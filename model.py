import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    # size of the hidden representation of each token
    # every token is represented by a vector of this size
    dim: int = 4096
    # number of layers in the transformer
    # each layer contains:
    # RMSNorm -> SelfAttention -> RMSNorm -> FeedForward
    n_layers: int = 32
    # number of heads for the queries
    n_heads: int = 32
    # number of heads for the key and values
    n_kv_heads: Optional[int] = None
    # the number of different tokens the model can understand
    # this is normally loaded from the model checkpoint
    vocab_size: int = -1
    # forces the hidden dimension to be a multiple of this number
    # useful for some hardware that requires the hidden dimension to be a multiple of 256
    # because the GPUs work more efficiently with certain matrix dimensions that are aligned to hardware friendly size such as 8, 16, 32, etc
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    # used to avoid division by zero in the RMSNorm layer
    norm_eps: float = 1e-5

    # Needed for KV Cache, maximum numbers needed for the cache
    # reserve enough kv cache memory for the maximum batch size and maximum sequence length
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None



def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # the dimension of the embedding must be even 
    assert head_dim % 2 == 0, "Dimensions must be divisible by 2"
    # build the theta parameters
    # according to the formula theta_i = 10000 ^ (-2i/dim) for i in range (0, dim - 1)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the positions (the m parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # using outer product to multiply each m be all elements in the vector theta
    # Shape: (seq_len) outer product* (head/2) -> (seq_len, head/2)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


### x is the tensor of a spacific token we want to apply rotary on it 
### freqs_complex is the tensor for the this token position
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # take two consecutive dimensions and group them
    # shape: (b, seq_len, H, head_dim) -> (b, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape: (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (b, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (b, seq_len, H, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (b, seq_len, H, head_dim / 2) -> (b, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (b, seq_len, H, head_dim / 2, 2) -> (b, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # repeat the keys and values to match the number of query heads
    # shape: (batch, seq_len, h_kv, head_dim) -> (batch, seq_len, h_kv * n_rep, head_dim)
    batch_size, seq_len, h_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    # (batch, seq_len, h_kv_heads, head_dim) -> (batch, seq_len, h_kv_heads, 1, head_dim) -> (batch, seq_len, h_kv_heads, n_rep, head_dim)
    # expand() creates a view that logically repeats the values
    # without allocating new memory for the repeated elements
    x = x.unsqueeze(3).expand(batch_size, seq_len, h_kv_heads, n_rep, head_dim)
    x = x.reshape(batch_size, seq_len, h_kv_heads * n_rep, head_dim)
    return x



class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        # one value for each hidden dimension, this is a learnable parameter that will be optimized during training
        # shape -> (dim) because every token entering RMSNorm will be represented by a vector of size dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # compute the root mean square of the input tensor along the last dimension
        # rsqrt = 1 / sqrt(...)
        # shape of x -> (batch, seq_len, dim) * (batch, seq_len, dim) -> (batch, seq_len, dim)
        # calculate the mean over the last dimension, to calculate the rms for each token independently
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # RMSNorm normalizes the magnitude of the representation.
        # The learnable weight allows the model to rescale each dimension.
        # to let the model learn how much should each dimension be scaled after normalization
        # shape of x -> (batch, seq_len, dim) * (dim) -> (batch, seq_len, dim)
        return self.weight * self._norm(x)



class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        # the number of heads for the key and values
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        # number of heads for the queries
        self.n_heads_q = args.n_heads
        # the repetition factor for the key and values heads to match the number of query heads
        # requires n_heads_q % n_kv_heads == 0
        assert self.n_heads_q % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # the dimension of each head
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.register_buffer(
            "cache_k", 
            torch.zeros(
              args.max_batch_size,
              args.max_seq_len,
              self.n_kv_heads,
              self.head_dim,
              device=args.device
            ),
            persistent=False
        )
        self.register_buffer(
            "cache_v", 
            torch.zeros(
              args.max_batch_size,
              args.max_seq_len,
              self.n_kv_heads,
              self.head_dim,
              device=args.device
            ),
            persistent=False
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (batch, 1, dim)

        # apply the wq, wk and wv matrices to queries, keys and values
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # divide the queries, keys and values into heads
        # (batch, 1, h_q * head_dim) -> (batch, 1, h_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # doesn't change the shape of the tensors
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)

        # replace the keys and values in the cache with the new ones
        # for all batches store the the new keys and values in the cache at the position of the current token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # retrieve the keys and values from the cache for all previous tokens up to the current token
        # shape -> (batch, seq_len, h_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # repeat the keys and values to match the number of query heads
        # shape -> (batch, seq_len, h_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # compute the attention scores
        # (batch, 1, h_q, head_dim) -> (batch, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # shape -> (batch, h_q, 1, head_dim) @ (batch, h_q, head_dim, seq_len_kv) -> (batch, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        out = torch.matmul(scores, values) # (batch, h_q, 1, seq_len_kv) @ (batch, h_q, seq_len_kv, head_dim) -> (batch, h_q, 1, head_dim)
        out = out.transpose(1, 2).contiguous() # (batch, 1, h_q, head_dim)
        out = out.view(batch_size, seq_len, -1) # (batch, 1, h_q * head_dim)
        out = self.wo(out) # (batch, 1, dim)
        return out



class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # the ffn gives the model more capacity to learn complex functions
        hidden_dim = 4 * args.dim
        # this is a design choice made by the llama architecture, to keep the hidden dimension as tranditional ffn
        # traditional ffn has 8 * dim^2  = 2 * dim * 4dim
        # swiglu has three matrices 3 * dim * hidden
        # so to keep the number of parameters similar to traditional ffn, we reduce the hidden dimension by 2/3
        hidden_dim = int(2 * hidden_dim / 3)
        # if the user specified a multiplier for the hidden dimension, use it
        if args.ffn_dim_multiplier is not None:
              hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        # round the hidden dimension to be a multiple of args.multiple_of
        hidden = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        return self.w2(swish * x_v)



class TransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization layers
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out



class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (n, theta) corresponding to the positions 
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
