"""
Common Layers for DeepSeek-V3

包含基础组件:
- RMSNorm
- RotaryEmbedding (RoPE)
- ExpertFFN (SwiGLU)

Reference: DeepSeek-V3 Technical Report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RMSNorm(nn.Module):
    """RMS Normalization
    
    比 LayerNorm 更高效，不需要计算均值和减均值操作
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            normalized: [batch, seq_len, hidden_size]
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码
    
    通过旋转矩阵将位置信息编码到 Query 和 Key 中
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 4096, 
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 计算逆频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, num_heads, head_dim] - 用于获取形状
            position_ids: [batch, seq_len]
        
        Returns:
            cos: [1, seq, 1, dim]
            sin: [1, seq, 1, dim]
        """
        # freqs: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", position_ids[0].float(), self.inv_freq)
        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转张量的一半维度
    
    将 [..., d] 拆分为 [..., d/2] 和 [..., d/2]，
    然后拼接为 [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:
    """应用 RoPE 到张量
    
    Args:
        x: [batch, seq, heads, dim] 或 [batch, heads, seq, dim]
        cos, sin: 位置编码
    
    Returns:
        rotated: 应用 RoPE 后的张量
    """
    return (x * cos) + (rotate_half(x) * sin)


class ExpertFFN(nn.Module):
    """单个专家的 FFN (SwiGLU 激活)
    
    SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size] 或 [num_tokens, hidden_size]
        Returns:
            output: 与输入形状相同
        """
        # SwiGLU activation
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
