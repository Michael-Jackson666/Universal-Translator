"""
DeepSeek-V3 Transformer Block

将 MLA 和 MoE 组合成完整的 Transformer Block

Reference: DeepSeek-V3 Technical Report
"""

import torch
import torch.nn as nn
from typing import Optional

from config import DeepSeekV3Config
from layers import RMSNorm
from mla import MultiHeadLatentAttention
from moe import DeepSeekMoE


class DeepSeekV3Block(nn.Module):
    """DeepSeek-V3 Transformer Block
    
    结构:
    1. RMSNorm -> MLA -> Residual
    2. RMSNorm -> MoE -> (MoE 内部有 Residual)
    """
    
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-Attention LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Multi-Head Latent Attention
        self.attention = MultiHeadLatentAttention(config, layer_idx)
        
        # Pre-FFN LayerNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoE FFN
        self.moe = DeepSeekMoE(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len]
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # ========== Attention Block ==========
        # Pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # MLA
        hidden_states = self.attention(hidden_states, position_ids, attention_mask)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # ========== MoE Block ==========
        # Pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MoE (内部已包含残差连接 u_t + shared + routed)
        hidden_states = self.moe(hidden_states)
        # 注意: MoE 内部的残差是加在 normalized 后的输入上
        # 这里我们需要保持一致性，由于 MoE 内部已加残差，直接返回
        
        return hidden_states
