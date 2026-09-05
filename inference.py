from typing import List, Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_batch_size: int, max_seq_len: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoints_dir}"
            chk_path = checkpoints[0]
            print(f"Loading model from {chk_path}")
            checkpoint = torch.load(chk_path, map_location=device)
            print(f"Loaded model in {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args)

        if load_model:
            # Remove rope.freqs from checkpoint to avoid loading it into the model
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model in {time.time() - prev_time:.2f} seconds")

        return LLaMA(model, tokenizer, model_args)

    ### the main generation loop. it takes text prompt, converts it to tokens, repeatedly asks llama "what token should come next"
    ### finally converts the generated token IDs back into text
    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        # max_seq_len -> loacal capacity including the prompt
        # max_gen_len -> how many tokens we want to generate
        if max_gen_len is None:
            # generate up to the maximum sequence length
            max_gen_len = self.model_args.max_seq_len - 1
        # convert prompts to token ids
        prompt_ids = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # make sure the batch size is not larger than the maximum batch size
        batch_size = len(prompt_ids)
        assert batch_size <= self.model_args.max_batch_size, f"Batch size {batch_size} is larger than the maximum batch size {self.model_args.max_batch_size}"
        max_prompt_len = max(len(ids) for ids in prompt_ids)
        # make sure the length of the prompt is not larger than the maximum sequence length
        assert max_prompt_len <= self.model_args.max_seq_len

        total_len = min(max_prompt_len + max_gen_len, self.model_args.max_seq_len)

        # create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        # initialize with the pad id 
        input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model_args.device)
        for i, ids in enumerate(prompt_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.model_args.device)

        # initialize with false for n prompts 
        eos_reached = torch.tensor([False] * batch_size, dtype=torch.bool, device=self.model_args.device)
        prompt_tokens_mask = input_ids != pad_id # True if the token is part of the prompt, False if it is padding or generated

        # start generation
        for cur_pos in tqdm(range(1, total_len), desc="Generating"):
            with torch.no_grad():
                # get the logits for the current position
                # shape -> (batch_size, 1, vocab_size)
                logits = self.model(input_ids[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # apply temperature and top-p sampling
                # process one token at a time (last one)
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # greedy decoding
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace the token if it is a padding token
            # if the position in part of the original prompt, keep it 
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], input_ids[:, cur_pos], next_token)
            input_ids[:, cur_pos] = next_token
            # ~prompt_tokens_mask[:, cur_pos] 
            # False -> original prompt token
            # True  -> generated-token position
            # we only care about EOS that the model generated
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if eos_reached.all():
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(input_ids.tolist()):
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]

            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)


    def _sample_top_p(self, probs, p):
        # probs_idx -> tell us what is the original position 
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_batch_size=4,
        max_seq_len=1024,
        device=device
    )

    print(f"Model loaded on {device} with max batch size {model.model_args.max_batch_size} and max seq len {model.model_args.max_seq_len}")

    # inference the model
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-' * 50)