"""
mHC Configuration

Reference: mHC: Manifold-Constrained Hyper-Connections (https://arxiv.org/abs/2512.24880)
"""

from dataclasses import dataclass


@dataclass
class mHCConfig:
    """mHC 配置类"""
    
    # 基础参数
    hidden_size: int = 2048           # C - 隐藏层维度
    num_streams: int = 4              # n - 残差流数量（扩展倍数）
    num_layers: int = 32              # L - 层数
    
    # Sinkhorn-Knopp 参数
    sinkhorn_iterations: int = 20     # t_max - 投影迭代次数
    
    # 动态重计算参数
    recompute_granularity: int = None  # L_r - 重计算分块大小，None 则自动计算
    
    # 缩放因子初始化
    alpha_pre_init: float = 1.0
    alpha_post_init: float = 1.0
    alpha_res_init: float = 1.0
    
    # RMSNorm
    rms_norm_eps: float = 1e-6
    
    def __post_init__(self):
        """验证配置并计算最优重计算分块"""
        assert self.num_streams >= 2, "n must be at least 2"
        
        # Eq. 20: 计算最优重计算分块大小
        if self.recompute_granularity is None:
            import math
            n = self.num_streams
            L = self.num_layers
            self.recompute_granularity = max(1, int(math.sqrt(n * L / (n + 2))))
    
    @property
    def stream_hidden_size(self) -> int:
        """宽残差流的总维度 n * C"""
        return self.num_streams * self.hidden_size
    
    @property
    def projection_out_dim(self) -> int:
        """投影输出维度 n^2 + 2n"""
        n = self.num_streams
        return n * n + 2 * n


def create_small_config() -> mHCConfig:
    """创建小型测试配置"""
    return mHCConfig(
        hidden_size=256,
        num_streams=4,
        num_layers=4,
        sinkhorn_iterations=10,
    )
