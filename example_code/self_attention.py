import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
	"""
	Basic Multi-Head Attention (no mask, no dropout, no positional encoding).

	Inputs/Outputs
	- Input: (batch, seq_len, embed_dim)
	- Output: (batch, seq_len, embed_dim)

	Args:
		embed_dim: input/output embedding dimension (D)
		num_heads: number of attention heads (H). Must divide embed_dim.
		bias: whether to use bias in the linear projections
	"""

	def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
		super().__init__()
		assert (
			embed_dim % num_heads == 0
		), "embed_dim must be divisible by num_heads"
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		# Separate projection layers for Q, K, V and output
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

	def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
		"""Split the last dimension into (num_heads, head_dim) and transpose.

		Input shape:  (B, T, D)
		Output shape: (B, H, T, Dh)
		"""
		B, T, _ = x.shape
		x = x.view(B, T, self.num_heads, self.head_dim)
		x = x.permute(0, 2, 1, 3)  # (B, H, T, Dh)
		return x

	def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
		"""Inverse of _split_heads.

		Input shape:  (B, H, T, Dh)
		Output shape: (B, T, D)
		"""
		B, H, T, Dh = x.shape
		x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, H, Dh)
		return x.view(B, T, H * Dh)

	def forward(
		self,
		query: torch.Tensor,
		key: Optional[torch.Tensor] = None,
		value: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute attention.

		If key/value are not provided, performs self-attention on query.

		Shapes:
			query: (B, Tq, D)
			key:   (B, Tk, D)
			value: (B, Tk, D)

		Returns:
			attn_out: (B, Tq, D)
			attn_weights: (B, H, Tq, Tk)
		"""
		if key is None:
			key = query
		if value is None:
			value = key

		# Linear projections
		Q = self.q_proj(query)  # (B, Tq, D)
		K = self.k_proj(key)    # (B, Tk, D)
		V = self.v_proj(value)  # (B, Tk, D)

		# Split into heads
		Q = self._split_heads(Q)  # (B, H, Tq, Dh)
		K = self._split_heads(K)  # (B, H, Tk, Dh)
		V = self._split_heads(V)  # (B, H, Tk, Dh)

		# Scaled dot-product attention
		scale = 1.0 / math.sqrt(self.head_dim)
		scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, Tq, Tk)
		attn_weights = F.softmax(scores, dim=-1)               # (B, H, Tq, Tk)
		attn = torch.matmul(attn_weights, V)                   # (B, H, Tq, Dh)

		# Combine heads and project
		attn = self._combine_heads(attn)  # (B, Tq, D)
		out = self.out_proj(attn)         # (B, Tq, D)
		return out, attn_weights


if __name__ == "__main__":
	# Minimal sanity check
	torch.manual_seed(0)
	B, T, D, H = 2, 5, 32, 4
	x = torch.randn(B, T, D)
	mha = MultiHeadAttention(embed_dim=D, num_heads=H)
	y, w = mha(x)  # self-attention

	print("Input shape:", x.shape)       # (2, 5, 32)
	print("Output shape:", y.shape)      # (2, 5, 32)
	print("Weights shape:", w.shape)     # (2, 4, 5, 5)

