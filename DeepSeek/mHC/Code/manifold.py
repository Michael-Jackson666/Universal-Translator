"""
mHC Manifold Projection

实现 Birkhoff 多面体投影和 Sinkhorn-Knopp 算法

Reference: mHC: Manifold-Constrained Hyper-Connections
"""

import torch
import torch.nn as nn


class SinkhornKnopp(nn.Module):
    """Sinkhorn-Knopp 迭代算法
    
    将任意正矩阵投影到双随机矩阵流形（Birkhoff 多面体）上
    
    双随机矩阵定义 (Eq. 6):
    - 每行和为 1: H @ 1_n = 1_n
    - 每列和为 1: 1_n^T @ H = 1_n^T  
    - 所有元素非负: H >= 0
    
    算法 (Eq. 9):
    M^(t) = T_r(T_c(M^(t-1)))
    其中 T_c 为列归一化，T_r 为行归一化
    """
    
    def __init__(self, num_iterations: int = 20, eps: float = 1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.eps = eps
    
    def forward(self, H_tilde: torch.Tensor) -> torch.Tensor:
        """将矩阵投影到双随机矩阵流形
        
        Args:
            H_tilde: [batch, n, n] 或 [n, n] - 未归一化的矩阵
        
        Returns:
            H_res: [batch, n, n] 或 [n, n] - 双随机矩阵
        """
        # Step 1: 指数映射确保正性
        # M^(0) = exp(H_tilde)
        M = torch.exp(H_tilde)
        
        # Step 2: Sinkhorn-Knopp 迭代 (Eq. 9)
        for _ in range(self.num_iterations):
            # T_c: 列归一化
            M = M / (M.sum(dim=-2, keepdim=True) + self.eps)
            # T_r: 行归一化
            M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
        
        return M


class ManifoldProjection(nn.Module):
    """流形投影模块
    
    实现 Eq. 8 的投影：
    - H_pre = sigmoid(H_tilde_pre)
    - H_post = 2 * sigmoid(H_tilde_post)
    - H_res = Sinkhorn-Knopp(H_tilde_res)
    """
    
    def __init__(self, num_streams: int, sinkhorn_iterations: int = 20):
        super().__init__()
        self.num_streams = num_streams
        self.sinkhorn = SinkhornKnopp(num_iterations=sinkhorn_iterations)
    
    def project_pre(self, H_tilde: torch.Tensor) -> torch.Tensor:
        """投影 Pre 映射 (Eq. 8)
        
        H_pre = sigmoid(H_tilde_pre)
        
        Sigmoid 保证非负性，防止正负抵消
        """
        return torch.sigmoid(H_tilde)
    
    def project_post(self, H_tilde: torch.Tensor) -> torch.Tensor:
        """投影 Post 映射 (Eq. 8)
        
        H_post = 2 * sigmoid(H_tilde_post)
        
        乘以 2 恢复残差的尺度
        """
        return 2.0 * torch.sigmoid(H_tilde)
    
    def project_res(self, H_tilde: torch.Tensor) -> torch.Tensor:
        """投影 Res 映射 (Eq. 8)
        
        H_res = Sinkhorn-Knopp(H_tilde_res)
        
        投影到双随机矩阵流形，保证：
        1. 信号守恒（凸组合）
        2. 范数有界（谱范数 <= 1）
        3. 组合封闭性
        """
        return self.sinkhorn(H_tilde)
    
    def forward(
        self, 
        H_tilde_pre: torch.Tensor,
        H_tilde_post: torch.Tensor,
        H_tilde_res: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """投影所有映射矩阵
        
        Args:
            H_tilde_pre: [batch, n] 或 [n]
            H_tilde_post: [batch, n] 或 [n]
            H_tilde_res: [batch, n, n] 或 [n, n]
        
        Returns:
            H_pre, H_post, H_res: 投影后的矩阵
        """
        H_pre = self.project_pre(H_tilde_pre)
        H_post = self.project_post(H_tilde_post)
        H_res = self.project_res(H_tilde_res)
        
        return H_pre, H_post, H_res
