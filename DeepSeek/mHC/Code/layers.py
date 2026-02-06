"""
mHC Layer Implementation

实现 mHC 的核心层，包含动态映射生成和流形投影

Reference: mHC: Manifold-Constrained Hyper-Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import mHCConfig
from manifold import ManifoldProjection


class RMSNorm(nn.Module):
    """RMS Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 注意：mHC 中 RMSNorm 的权重被融合到投影矩阵中
        # 这里保留标准实现用于理解
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size]
        Returns:
            normalized: [..., hidden_size]
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class mHCDynamicMapping(nn.Module):
    """mHC 动态映射生成器
    
    实现 Eq. 7: 从输入生成未归一化的映射矩阵
    
    Step 1: 线性投影生成原始参数
    - H_tilde_pre = alpha_pre * (x' @ phi_pre) + b_pre
    - H_tilde_post = alpha_post * (x' @ phi_post) + b_post  
    - H_tilde_res = alpha_res * mat(x' @ phi_res) + b_res
    
    其中 x' = RMSNorm(vec(x))
    """
    
    def __init__(self, config: mHCConfig):
        super().__init__()
        self.config = config
        n = config.num_streams
        nC = config.stream_hidden_size
        
        # 融合投影矩阵 phi (Eq. 10)
        # phi: [nC, n^2 + 2n]
        self.phi = nn.Parameter(torch.randn(nC, n * n + 2 * n) * 0.02)
        
        # 缩放因子 alpha (Eq. 12)
        self.alpha_pre = nn.Parameter(torch.tensor(config.alpha_pre_init))
        self.alpha_post = nn.Parameter(torch.tensor(config.alpha_post_init))
        self.alpha_res = nn.Parameter(torch.tensor(config.alpha_res_init))
        
        # 偏置 b (Eq. 13)
        # [n^2 + 2n]
        self.bias = nn.Parameter(torch.zeros(n * n + 2 * n))
        
        # RMSNorm eps
        self.eps = config.rms_norm_eps
        self.n = n
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从输入生成动态映射矩阵
        
        Args:
            x: [batch, n, C] - 宽残差流输入
        
        Returns:
            H_tilde_pre: [batch, n]
            H_tilde_post: [batch, n]
            H_tilde_res: [batch, n, n]
        """
        batch_size = x.shape[0]
        n = self.n
        
        # Step 1: 展平为向量 vec(x) (Eq. 7)
        # x: [batch, n, C] -> [batch, nC]
        x_vec = x.view(batch_size, -1)
        
        # Step 2: 计算 RMS 归一化因子 r (Eq. 15)
        # r = ||x||_2 / sqrt(nC)
        r = torch.sqrt(x_vec.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # Step 3: 融合线性投影 (Eq. 14)
        # [batch, nC] @ [nC, n^2+2n] -> [batch, n^2+2n]
        H_tilde_all = x_vec @ self.phi
        
        # Step 4: 应用 RMSNorm 和缩放 (Eq. 16)
        # 分割为 pre, post, res
        H_tilde_pre = self.alpha_pre * H_tilde_all[:, :n] / r + self.bias[:n]
        H_tilde_post = self.alpha_post * H_tilde_all[:, n:2*n] / r + self.bias[n:2*n]
        H_tilde_res_flat = self.alpha_res * H_tilde_all[:, 2*n:] / r + self.bias[2*n:]
        
        # Step 5: 重构 res 矩阵 mat(·)
        # [batch, n^2] -> [batch, n, n]
        H_tilde_res = H_tilde_res_flat.view(batch_size, n, n)
        
        return H_tilde_pre, H_tilde_post, H_tilde_res


class mHCLayer(nn.Module):
    """mHC 层
    
    实现完整的 mHC 前向传播 (Eq. 3):
    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    
    其中映射矩阵通过动态生成 + 流形投影得到
    """
    
    def __init__(
        self, 
        config: mHCConfig,
        layer_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.n = config.num_streams
        self.C = config.hidden_size
        
        # 动态映射生成器
        self.dynamic_mapping = mHCDynamicMapping(config)
        
        # 流形投影
        self.manifold_proj = ManifoldProjection(
            num_streams=config.num_streams,
            sinkhorn_iterations=config.sinkhorn_iterations,
        )
        
        # 内部层函数 F(·)
        # 如果未提供，使用简单的 FFN
        if layer_fn is None:
            self.layer_fn = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.GELU(),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
            )
        else:
            self.layer_fn = layer_fn
    
    def _aggregate_streams(
        self, 
        x: torch.Tensor, 
        H_pre: torch.Tensor
    ) -> torch.Tensor:
        """聚合宽残差流为层输入
        
        Args:
            x: [batch, n, C] - 宽残差流
            H_pre: [batch, n] - Pre 映射向量
        
        Returns:
            aggregated: [batch, C]
        """
        # H_pre @ x: 加权求和
        # [batch, n] @ [batch, n, C] -> [batch, C]
        # 使用 einsum: batch 维度广播
        return torch.einsum('bn,bnc->bc', H_pre, x)
    
    def _broadcast_to_streams(
        self, 
        y: torch.Tensor, 
        H_post: torch.Tensor
    ) -> torch.Tensor:
        """将层输出广播回宽残差流
        
        Args:
            y: [batch, C] - 层输出
            H_post: [batch, n] - Post 映射向量
        
        Returns:
            broadcasted: [batch, n, C]
        """
        # H_post^T @ y: 广播
        # [batch, n, 1] * [batch, 1, C] -> [batch, n, C]
        return H_post.unsqueeze(-1) * y.unsqueeze(1)
    
    def _mix_streams(
        self, 
        x: torch.Tensor, 
        H_res: torch.Tensor
    ) -> torch.Tensor:
        """残差流混合
        
        Args:
            x: [batch, n, C] - 输入残差流
            H_res: [batch, n, n] - 双随机混合矩阵
        
        Returns:
            mixed: [batch, n, C]
        """
        # H_res @ x
        # [batch, n, n] @ [batch, n, C] -> [batch, n, C]
        return torch.einsum('bmn,bnc->bmc', H_res, x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """mHC 层前向传播
        
        Eq. 3: x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
        
        Args:
            x: [batch, n, C] - 宽残差流输入
        
        Returns:
            x_out: [batch, n, C] - 宽残差流输出
        """
        # Step 1: 动态生成映射矩阵 (Eq. 7)
        H_tilde_pre, H_tilde_post, H_tilde_res = self.dynamic_mapping(x)
        
        # Step 2: 流形投影 (Eq. 8, 17-19)
        H_pre, H_post, H_res = self.manifold_proj(
            H_tilde_pre, H_tilde_post, H_tilde_res
        )
        
        # Step 3: 聚合 (H_pre @ x_l)
        layer_input = self._aggregate_streams(x, H_pre)
        
        # Step 4: 层计算 F(·)
        layer_output = self.layer_fn(layer_input)
        
        # Step 5: 广播 (H_post^T @ F(...))
        broadcast_output = self._broadcast_to_streams(layer_output, H_post)
        
        # Step 6: 残差混合 (H_res @ x_l)
        residual = self._mix_streams(x, H_res)
        
        # Eq. 3: 相加
        x_out = residual + broadcast_output
        
        return x_out
