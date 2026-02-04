"""
DeepSeek-V3 Complete Model Implementation

主模型和语言模型头的实现

Reference: DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437)

总参数量: 671B
激活参数量: 37B (per token)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from config import DeepSeekV3Config
from layers import RMSNorm
from block import DeepSeekV3Block
from mtp import MultiTokenPrediction


class DeepSeekV3Model(nn.Module):
    """
    DeepSeek-V3 主模型 (不含 LM Head)
    
    结构:
    1. Token Embedding
    2. N 个 DeepSeekV3Block (MLA + MoE)
    3. Final RMSNorm
    """
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            DeepSeekV3Block(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Final LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """准备 causal attention mask
        
        Args:
            attention_mask: [batch, seq_len] - 1 表示有效, 0 表示 padding
        
        Returns:
            causal_mask: [batch, 1, seq_len, seq_len]
        """
        # Causal mask: 上三角为 -inf
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        
        # 结合 padding mask (如果提供)
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq]
            extended_mask = attention_mask[:, None, None, :]
            # 0 位置设为 -inf
            extended_mask = (1.0 - extended_mask) * float("-inf")
            causal_mask = causal_mask + extended_mask
        
        return causal_mask
    
    def _prepare_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """准备 position IDs"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        return position_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        position_ids = self._prepare_position_ids(input_ids, position_ids)
        
        # Token Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Attention Mask
        causal_mask = self._prepare_attention_mask(
            attention_mask, batch_size, seq_len, device, hidden_states.dtype
        )
        
        # Transformer Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, causal_mask)
        
        # Final Norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class DeepSeekV3ForCausalLM(nn.Module):
    """DeepSeek-V3 语言模型（用于生成）
    
    包含:
    1. DeepSeekV3Model (Transformer)
    2. LM Head (共享 embedding 权重可选)
    3. MTP 模块 (可选)
    """
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        
        # 主模型
        self.model = DeepSeekV3Model(config)
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # MTP 模块 (可选)
        self.mtp = None
        if config.mtp_depth > 0:
            self.mtp = MultiTokenPrediction(
                config,
                embed_tokens=self.model.embed_tokens,
                lm_head=self.lm_head,
                transformer_block_class=DeepSeekV3Block,
            )
    
    def _compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算语言模型损失 (交叉熵)"""
        # Shift: 预测下一个 token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_mtp_logits: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] - 用于计算损失
            return_mtp_logits: 是否返回 MTP logits
        
        Returns:
            loss: 总损失 (如果提供 labels)
            logits: [batch, seq_len, vocab_size]
            mtp_logits: List (可选)
        """
        batch_size, seq_len = input_ids.shape
        
        # Position IDs
        position_ids = self.model._prepare_position_ids(input_ids)
        
        # 主模型前向传播
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        # MTP
        mtp_logits = None
        if (return_mtp_logits or labels is not None) and self.mtp is not None:
            mtp_logits = self.mtp(input_ids, hidden_states, position_ids)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 主模型损失
            loss = self._compute_lm_loss(logits, labels)
            
            # MTP 损失
            if mtp_logits is not None and len(mtp_logits) > 0:
                mtp_loss = self.mtp.compute_loss(
                    mtp_logits, 
                    labels, 
                    self.config.vocab_size,
                    mtp_weight=0.3  # λ
                )
                loss = loss + mtp_loss
        
        return loss, logits, mtp_logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """自回归生成
        
        Args:
            input_ids: [batch, seq_len] - 输入 prompt
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
        
        Returns:
            output_ids: [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # 前向传播
            _, logits, _ = self.forward(input_ids)
            
            # 取最后一个位置的 logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-P 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.model.embed_tokens.parameters())
        attention = sum(
            sum(p.numel() for p in layer.attention.parameters())
            for layer in self.model.layers
        )
        moe = sum(
            sum(p.numel() for p in layer.moe.parameters())
            for layer in self.model.layers
        )
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        
        return {
            "total": total,
            "embedding": embedding,
            "attention": attention,
            "moe": moe,
            "lm_head": lm_head,
        }
