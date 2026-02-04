"""
DeepSeek-V3 Configuration

Reference: DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437)
"""

from dataclasses import dataclass


@dataclass
class DeepSeekV3Config:
    """DeepSeek-V3 配置"""
    
    # 基础参数
    vocab_size: int = 102400
    hidden_size: int = 7168          # d
    intermediate_size: int = 18432   # FFN 中间层
    num_layers: int = 61
    
    # MLA 参数
    num_attention_heads: int = 128   # n_h
    head_dim: int = 128              # d_h
    kv_compress_dim: int = 512       # d_c (KV 压缩维度)
    q_compress_dim: int = 1536       # d'_c (Query 压缩维度)
    rope_dim: int = 64               # d_h^R (RoPE 解耦维度)
    
    # MoE 参数
    num_shared_experts: int = 1      # N_s
    num_routed_experts: int = 256    # N_r
    num_active_experts: int = 8      # K_r
    expert_intermediate_size: int = 2048
    
    # MTP 参数
    mtp_depth: int = 1               # D (预测深度)
    
    # 其他
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        """验证配置"""
        assert self.num_active_experts <= self.num_routed_experts
        assert self.kv_compress_dim < self.num_attention_heads * self.head_dim
        assert self.q_compress_dim < self.num_attention_heads * self.head_dim


def create_small_config() -> DeepSeekV3Config:
    """创建小型测试配置"""
    return DeepSeekV3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=2,
        num_attention_heads=8,
        head_dim=32,
        kv_compress_dim=64,
        q_compress_dim=128,
        rope_dim=16,
        num_shared_experts=1,
        num_routed_experts=8,
        num_active_experts=2,
        expert_intermediate_size=128,
        mtp_depth=1,
        max_position_embeddings=512,
    )
