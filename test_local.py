import json
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from model import ModelArgs, Transformer
from dummy_tokenizer import DummyTokenizer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: DummyTokenizer, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build(checkpoints_dir: str, max_batch_size: int, max_seq_len: int, device: str):
        """
        Builds a small LLaMA model with RANDOM weights and a DummyTokenizer.
        No .pth checkpoint file is needed - this is only for testing the
        model code (shapes, forward pass, generation loop), not real output quality.
        """
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)

        tokenizer = DummyTokenizer()

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )
        model_args.vocab_size = tokenizer.vocab_size

        # Keep tensors as normal float32 for a CPU test run
        torch.set_default_dtype(torch.float32)

        model = Transformer(model_args).to(device)
        model.eval()

        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1

        prompt_ids = [self.tokenizer.encode(prompt, add_bos=True, add_eos=False) for prompt in prompts]

        batch_size = len(prompt_ids)
        assert batch_size <= self.model_args.max_batch_size, \
            f"Batch size {batch_size} is larger than the maximum batch size {self.model_args.max_batch_size}"
        max_prompt_len = max(len(ids) for ids in prompt_ids)
        assert max_prompt_len <= self.model_args.max_seq_len

        total_len = min(max_prompt_len + max_gen_len, self.model_args.max_seq_len)

        pad_id = self.tokenizer.pad_id()
        input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model_args.device)
        for i, ids in enumerate(prompt_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.model_args.device)

        eos_reached = torch.tensor([False] * batch_size, dtype=torch.bool, device=self.model_args.device)
        prompt_tokens_mask = input_ids != pad_id

        for cur_pos in tqdm(range(1, total_len), desc="Generating"):
            with torch.no_grad():
                logits = self.model(input_ids[:, cur_pos - 1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], input_ids[:, cur_pos], next_token)
            input_ids[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if eos_reached.all():
                break

        out_tokens = []
        out_text = []
        for current_prompt_tokens in input_ids.tolist():
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return out_tokens, out_text

    def _sample_top_p(self, probs, p):
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
    device = "cpu"

    prompts = [
        "hello world",
        "testing a tiny model",
    ]

    prev_time = time.time()
    model = LLaMA.build(
        checkpoints_dir=".",
        max_batch_size=4,
        max_seq_len=32,
        device=device,
    )
    print(f"Random test model built in {time.time() - prev_time:.2f} seconds")
    print(f"vocab_size={model.model_args.vocab_size}, dim={model.model_args.dim}, "
          f"n_layers={model.model_args.n_layers}, n_heads={model.model_args.n_heads}")

    out_tokens, out_text = model.text_completion(prompts, max_gen_len=10)
    for i in range(len(out_text)):
        print(f"Prompt: {prompts[i]!r}")
        print(f"Generated token ids: {out_text[i]}")
        print("-" * 50)