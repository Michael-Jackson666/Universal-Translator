"""
Multi-Token Prediction (MTP) for DeepSeek-V3

用于预测额外的 D 个 token：
1. 训练时提升数据效率
2. 推理时的投机解码 (Speculative Decoding)

Reference: DeepSeek-V3 Technical Report, Section 4
"""

import torch
import torch.nn as nn
from typing import Optional

from config import DeepSeekV3Config
from layers import RMSNorm


class MultiTokenPredictionModule(nn.Module):
    """
    Multi-Token Prediction (MTP) 模块
    
    第 k 个 MTP 模块由以下组成：
    - 共享嵌入层 Emb(·)
    - 共享输出头 OutHead(·)
    - Transformer 块 TRM_k(·)
    - 投影矩阵 M_k
    """
    
    def __init__(
        self, 
        config: DeepSeekV3Config, 
        depth: int,
        transformer_block_class,  # 传入 DeepSeekV3Block 类
    ):
        super().__init__()
        self.depth = depth
        self.hidden_size = config.hidden_size
        
        # 投影矩阵 M_k (Eq. 21)
        # M_k: 2d -> d
        self.projection = nn.Linear(
            2 * config.hidden_size, 
            config.hidden_size, 
            bias=False
        )
        
        # RMSNorm for representations and embeddings
        self.norm_repr = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_emb = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Transformer Block (TRM_k)
        # 使用较大的 layer_idx 以区分
        self.transformer = transformer_block_class(config, layer_idx=1000 + depth)
    
    def forward(
        self,
        prev_hidden: torch.Tensor,      # h_i^{k-1}
        next_token_emb: torch.Tensor,   # Emb(t_{i+k})
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prev_hidden: [batch, seq_len, hidden_size] - 上一层的表示
            next_token_emb: [batch, seq_len, hidden_size] - 下一个 token 的 embedding
            position_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len]
        
        Returns:
            hidden: [batch, seq_len, hidden_size] - 当前层的表示 h_i^k
        """
        # Eq. 21: h'_i^k = M_k [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]
        concat_input = torch.cat([
            self.norm_repr(prev_hidden),
            self.norm_emb(next_token_emb)
        ], dim=-1)  # [batch, seq, 2*hidden]
        
        hidden = self.projection(concat_input)  # [batch, seq, hidden]
        
        # Eq. 22: h_{1:T-k}^k = TRM_k(h'_{1:T-k})
        hidden = self.transformer(hidden, position_ids, attention_mask)
        
        return hidden


class MultiTokenPrediction(nn.Module):
    """
    Multi-Token Prediction 管理器
    
    管理 D 个 MTP 模块，计算 MTP 损失
    """
    
    def __init__(
        self, 
        config: DeepSeekV3Config,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
        transformer_block_class,
    ):
        super().__init__()
        self.config = config
        self.mtp_depth = config.mtp_depth
        
        # 共享的 embedding 和 lm_head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        
        # D 个 MTP 模块
        self.mtp_modules = nn.ModuleList([
            MultiTokenPredictionModule(config, depth=k+1, transformer_block_class=transformer_block_class)
            for k in range(self.mtp_depth)
        ])
    
    def _prepare_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """准备 causal attention mask"""
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1
        )
        return causal_mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """计算所有 MTP 模块的 logits
        
        Args:
            input_ids: [batch, seq_len] - 原始输入 token
            hidden_states: [batch, seq_len, hidden_size] - 主模型的输出
            position_ids: [batch, seq_len]
        
        Returns:
            mtp_logits: List of [batch, seq_len-k, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = hidden_states.dtype
        
        mtp_logits = []
        prev_hidden = hidden_states
        
        for k, mtp_module in enumerate(self.mtp_modules):
            # 获取 t_{i+k} 的 embedding (shifted)
            if k + 1 < seq_len:
                shifted_ids = input_ids[:, k+1:]  # [batch, seq-k-1]
                next_token_emb = self.embed_tokens(shifted_ids)
                
                # 截断 prev_hidden 以匹配
                prev_hidden_truncated = prev_hidden[:, :seq_len-k-1, :]
                position_ids_truncated = position_ids[:, :seq_len-k-1]
                
                # 更新 causal mask
                truncated_mask = self._prepare_attention_mask(
                    batch_size, 
                    seq_len - k - 1, 
                    device, 
                    dtype
                )
                
                # Eq. 21, 22: MTP 前向传播
                mtp_hidden = mtp_module(
                    prev_hidden_truncated,
                    next_token_emb,
                    position_ids_truncated,
                    truncated_mask
                )
                
                # Eq. 23: P_{i+k+1}^k = OutHead(h_i^k)
                mtp_logit = self.lm_head(mtp_hidden)
                mtp_logits.append(mtp_logit)
                
                # 更新 prev_hidden 为下一个 MTP 模块
                prev_hidden = mtp_hidden
        
        return mtp_logits
    
    def compute_loss(
        self,
        mtp_logits: list[torch.Tensor],
        labels: torch.Tensor,
        vocab_size: int,
        mtp_weight: float = 0.3,
    ) -> torch.Tensor:
        """计算 MTP 损失
        
        Eq. 24: L_MTP^k = CrossEntropy(P_{2+k:T+1}^k, t_{2+k:T+1})
        Eq. 25: L_MTP = (λ/D) * Σ_{k=1}^D L_MTP^k
        
        Args:
            mtp_logits: List of [batch, seq_len-k, vocab_size]
            labels: [batch, seq_len]
            vocab_size: 词表大小
            mtp_weight: λ 权重
        
        Returns:
            loss: MTP 总损失
        """
        import torch.nn.functional as F
        
        total_loss = 0.0
        
        for k, mtp_logit in enumerate(mtp_logits):
            # 对齐 labels: 需要预测 t_{k+2:T+1}
            mtp_shift_labels = labels[:, k+2:].contiguous()
            
            if mtp_shift_labels.numel() > 0:
                # 计算交叉熵
                mtp_loss = F.cross_entropy(
                    mtp_logit[:, :-1, :].contiguous().view(-1, vocab_size),
                    mtp_shift_labels[:, :-1].contiguous().view(-1),
                    ignore_index=-100
                )
                total_loss = total_loss + mtp_loss
        
        # Eq. 25: 平均并加权
        if len(mtp_logits) > 0:
            total_loss = mtp_weight * total_loss / len(mtp_logits)
        
        return total_loss
