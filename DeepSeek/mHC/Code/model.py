"""
mHC Model Implementation

完整的 mHC 模型，包含动态重计算支持

Reference: mHC: Manifold-Constrained Hyper-Connections
"""

import torch
import torch.nn as nn
from typing import Optional, List

from config import mHCConfig
from layers import mHCLayer, RMSNorm


class mHCModel(nn.Module):
    """mHC 完整模型
    
    特点：
    1. 宽残差流 (n × C)
    2. 流形约束保证稳定性
    3. 动态重计算节省显存
    """
    
    def __init__(self, config: mHCConfig):
        super().__init__()
        self.config = config
        self.n = config.num_streams
        self.C = config.hidden_size
        
        # 输入投影: C -> n × C
        self.input_proj = nn.Linear(config.hidden_size, config.stream_hidden_size)
        
        # mHC 层
        self.layers = nn.ModuleList([
            mHCLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # 输出聚合: n × C -> C
        self.output_proj = nn.Linear(config.stream_hidden_size, config.hidden_size)
        
        # Final Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def _expand_to_streams(self, x: torch.Tensor) -> torch.Tensor:
        """将标准输入扩展为宽残差流
        
        Args:
            x: [batch, C]
        
        Returns:
            x_wide: [batch, n, C]
        """
        batch_size = x.shape[0]
        # 通过投影扩展
        x_projected = self.input_proj(x)  # [batch, nC]
        return x_projected.view(batch_size, self.n, self.C)
    
    def _collapse_streams(self, x: torch.Tensor) -> torch.Tensor:
        """将宽残差流聚合为标准输出
        
        Args:
            x: [batch, n, C]
        
        Returns:
            x_collapsed: [batch, C]
        """
        batch_size = x.shape[0]
        # 展平后投影
        x_flat = x.view(batch_size, -1)  # [batch, nC]
        return self.output_proj(x_flat)  # [batch, C]
    
    def forward(
        self, 
        x: torch.Tensor,
        use_recompute: bool = False,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch, C] - 标准维度输入
            use_recompute: 是否使用动态重计算（节省显存）
        
        Returns:
            output: [batch, C]
        """
        # 扩展到宽残差流
        hidden_states = self._expand_to_streams(x)
        
        if use_recompute:
            # 动态重计算模式
            hidden_states = self._forward_with_recompute(hidden_states)
        else:
            # 标准前向传播
            for layer in self.layers:
                hidden_states = layer(hidden_states)
        
        # 聚合输出
        output = self._collapse_streams(hidden_states)
        output = self.norm(output)
        
        return output
    
    def _forward_with_recompute(self, x: torch.Tensor) -> torch.Tensor:
        """带动态重计算的前向传播
        
        Eq. 20: L_r* = sqrt(nL / (n+2))
        
        策略：
        - 每 L_r 层保存一个 checkpoint
        - 反向传播时重新计算中间层
        """
        L_r = self.config.recompute_granularity
        
        # 使用 PyTorch 的 checkpoint 机制
        from torch.utils.checkpoint import checkpoint_sequential
        
        # 将层分成多个段
        num_segments = max(1, len(self.layers) // L_r)
        
        x = checkpoint_sequential(
            self.layers,
            segments=num_segments,
            input=x,
            use_reentrant=False,
        )
        
        return x


class mHCForSequenceModeling(nn.Module):
    """用于序列建模的 mHC 模型
    
    将 mHC 应用于序列的每个位置
    """
    
    def __init__(self, config: mHCConfig, seq_len: int = 512):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        
        # 位置编码
        self.position_embedding = nn.Embedding(seq_len, config.hidden_size)
        
        # mHC 模型
        self.mhc = mHCModel(config)
    
    def forward(
        self, 
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, C]
            position_ids: [batch, seq_len]
        
        Returns:
            output: [batch, seq_len, C]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(position_ids)
        x = x + pos_emb
        
        # 对每个位置应用 mHC
        # 可以并行处理
        x_flat = x.view(batch_size * seq_len, hidden_size)
        output_flat = self.mhc(x_flat)
        output = output_flat.view(batch_size, seq_len, hidden_size)
        
        return output
