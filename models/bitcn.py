"""BiTCN -- Bidirectional Temporal Convolutional Network.

Set causal=True for online mode (left-only padding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import register


class DilatedResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.0, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.pad = (kernel - 1) * dilation if causal else ((kernel - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.dropout = nn.Dropout(dropout)

    def _do_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            return F.pad(x, (self.pad, 0))
        return F.pad(x, (self.pad, self.pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(self._do_pad(x))))
        h = self.dropout(h)
        h = self.norm2(self.conv2(self._do_pad(h)))
        return F.gelu(x + h)


@register("bitcn")
class BiTCN(nn.Module):
    """(B, n_in, T) -> (B, n_out, T)"""

    def __init__(self, n_in: int, n_out: int, channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
                 kernel: int = 3, dropout: float = 0.05,
                 causal: bool = False) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(n_in, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            DilatedResBlock(channels, kernel=kernel, dilation=d,
                            dropout=dropout, causal=causal)
            for d in dilations])
        self.output_proj = nn.Conv1d(channels, n_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)
