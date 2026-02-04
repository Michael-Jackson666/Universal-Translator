"""
Multi-Head Latent Attention (MLA) for DeepSeek-V3

核心创新：
1. KV 低秩联合压缩：不缓存完整 KV，而是缓存压缩潜变量 c_KV
2. 解耦 RoPE：位置编码单独处理，不影响压缩
3. Query 压缩：减少训练时激活显存

Reference: DeepSeek-V3 Technical Report, Section 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from config import DeepSeekV3Config
from layers import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    
    相比标准 MHA，MLA 的 KV Cache 显著减少：
    - MHA: 缓存 2 * n_h * d_h * seq_len
    - MLA: 缓存 (d_c + d_h^R) * seq_len
    
    当 d_c << n_h * d_h 时，显存节省显著
    """
    
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_compress_dim = config.kv_compress_dim
        self.q_compress_dim = config.q_compress_dim
        self.rope_dim = config.rope_dim
        
        # ============ KV 压缩投影 ============
        # Eq. 1: W^{DKV}: d -> d_c
        self.W_DKV = nn.Linear(self.hidden_size, self.kv_compress_dim, bias=False)
        
        # Eq. 2: W^{UK}: d_c -> n_h * d_h
        self.W_UK = nn.Linear(self.kv_compress_dim, self.num_heads * self.head_dim, bias=False)
        
        # Eq. 5: W^{UV}: d_c -> n_h * d_h
        self.W_UV = nn.Linear(self.kv_compress_dim, self.num_heads * self.head_dim, bias=False)
        
        # Eq. 3: W^{KR}: d -> d_h^R (所有头共享的 RoPE Key)
        self.W_KR = nn.Linear(self.hidden_size, self.rope_dim, bias=False)
        
        # ============ Query 压缩投影 ============
        # Eq. 6: W^{DQ}: d -> d'_c
        self.W_DQ = nn.Linear(self.hidden_size, self.q_compress_dim, bias=False)
        
        # Eq. 7: W^{UQ}: d'_c -> n_h * d_h
        self.W_UQ = nn.Linear(self.q_compress_dim, self.num_heads * self.head_dim, bias=False)
        
        # Eq. 8: W^{QR}: d'_c -> n_h * d_h^R
        self.W_QR = nn.Linear(self.q_compress_dim, self.num_heads * self.rope_dim, bias=False)
        
        # ============ 输出投影 ============
        # Eq. 11: W^O: n_h * d_h -> d
        self.W_O = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Attention 缩放因子 (使用拼接后的完整维度)
        self.scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)
    
    def _compute_kv(
        self, 
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 KV 压缩表示
        
        Args:
            hidden_states: [batch, seq, hidden_size]
        
        Returns:
            c_KV: [batch, seq, d_c] - 压缩的 KV 潜变量 (用于缓存)
            k_C: [batch, seq, n_h, d_h] - 内容 Key
            k_R: [batch, seq, 1, d_h^R] - RoPE Key (用于缓存)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Eq. 1: c_t^{KV} = W^{DKV} h_t
        c_KV = self.W_DKV(hidden_states)  # [batch, seq, d_c]
        
        # Eq. 2: k_t^C = W^{UK} c_t^{KV}
        k_C = self.W_UK(c_KV)  # [batch, seq, n_h * d_h]
        k_C = k_C.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Eq. 3: k_t^R = W^{KR} h_t (RoPE 稍后应用)
        k_R = self.W_KR(hidden_states)  # [batch, seq, d_h^R]
        k_R = k_R.unsqueeze(2)  # [batch, seq, 1, d_h^R]
        
        return c_KV, k_C, k_R
    
    def _compute_value(self, c_KV: torch.Tensor) -> torch.Tensor:
        """从压缩表示计算 Value
        
        Args:
            c_KV: [batch, seq, d_c]
        
        Returns:
            v_C: [batch, seq, n_h, d_h]
        """
        batch_size, seq_len, _ = c_KV.shape
        
        # Eq. 5: v_t^C = W^{UV} c_t^{KV}
        v_C = self.W_UV(c_KV)  # [batch, seq, n_h * d_h]
        v_C = v_C.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        return v_C
    
    def _compute_query(
        self, 
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 Query 压缩表示
        
        Args:
            hidden_states: [batch, seq, hidden_size]
        
        Returns:
            q_C: [batch, seq, n_h, d_h] - 内容 Query
            q_R: [batch, seq, n_h, d_h^R] - RoPE Query
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Eq. 6: c_t^Q = W^{DQ} h_t
        c_Q = self.W_DQ(hidden_states)  # [batch, seq, d'_c]
        
        # Eq. 7: q_t^C = W^{UQ} c_t^Q
        q_C = self.W_UQ(c_Q)  # [batch, seq, n_h * d_h]
        q_C = q_C.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Eq. 8: q_t^R = W^{QR} c_t^Q (RoPE 稍后应用)
        q_R = self.W_QR(c_Q)  # [batch, seq, n_h * d_h^R]
        q_R = q_R.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        
        return q_C, q_R
    
    def _apply_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算注意力
        
        Args:
            q: [batch, n_h, seq, d_h + d_h^R]
            k: [batch, n_h, seq, d_h + d_h^R]
            v: [batch, n_h, seq, d_h]
            attention_mask: [batch, 1, seq, seq]
        
        Returns:
            output: [batch, seq, n_h * d_h]
        """
        batch_size = q.shape[0]
        seq_len = q.shape[2]
        
        # Eq. 10: attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Eq. 10: o_{t,i} = sum_j softmax(...) * v_{j,i}^C
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_h, seq, d_h]
        
        # Reshape: [batch, n_h, seq, d_h] -> [batch, seq, n_h * d_h]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] - h_t
            position_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, hidden_size] - u_t
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Step 1: KV 压缩与生成
        c_KV, k_C, k_R = self._compute_kv(hidden_states)
        v_C = self._compute_value(c_KV)
        
        # Step 2: Query 压缩与生成
        q_C, q_R = self._compute_query(hidden_states)
        
        # Step 3: 应用 RoPE
        cos, sin = self.rotary_emb(q_R, position_ids)
        
        q_R = apply_rotary_pos_emb(q_R, cos, sin)
        # k_R 需要广播到所有头
        k_R_expanded = k_R.expand(-1, -1, self.num_heads, -1)
        k_R_expanded = apply_rotary_pos_emb(k_R_expanded, cos, sin)
        
        # Step 4: 拼接得到完整的 Q 和 K
        # Eq. 4, 9: q_{t,i} = [q_{t,i}^C; q_{t,i}^R], k_{t,i} = [k_{t,i}^C; k_t^R]
        q = torch.cat([q_C, q_R], dim=-1)  # [batch, seq, n_h, d_h + d_h^R]
        k = torch.cat([k_C, k_R_expanded], dim=-1)  # [batch, seq, n_h, d_h + d_h^R]
        
        # 转置为 [batch, n_h, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v_C = v_C.transpose(1, 2)
        
        # Step 5: Attention 计算
        attn_output = self._apply_attention(q, k, v_C, attention_mask)
        
        # Eq. 11: u_t = W^O [o_{t,1}; ...; o_{t,n_h}]
        output = self.W_O(attn_output)
        
        return output
