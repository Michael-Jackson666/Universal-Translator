"""
DeepSeek MoE Layer for DeepSeek-V3

核心特点：
1. 细粒度专家 (Fine-grained Experts) - 更多但更小的专家
2. 共享专家 (Shared Experts) - 所有 token 都经过
3. 无辅助损失负载均衡 (Auxiliary-Loss-Free Load Balancing)

Reference: DeepSeek-V3 Technical Report, Section 3
"""

import torch
import torch.nn as nn
from typing import Optional

from config import DeepSeekV3Config
from layers import ExpertFFN


class ExpertRouter(nn.Module):
    """专家路由器
    
    实现无辅助损失负载均衡 (Auxiliary-Loss-Free Load Balancing)
    
    关键点：
    - Bias b_i 仅用于决定选哪个专家
    - 最终权重使用原始亲和度分数 s_{i,t}
    """
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_active_experts = config.num_active_experts
        
        # 专家质心向量 e_i (Eq. 15)
        self.expert_centroids = nn.Parameter(
            torch.randn(self.num_routed_experts, config.hidden_size) * 0.02
        )
        
        # 负载均衡 Bias b_i (Eq. 16) - 仅用于路由选择，不参与梯度
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.num_routed_experts)
        )
    
    def compute_affinity_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算亲和度分数
        
        Eq. 15: s_{i,t} = Sigmoid(u_t^T e_i)
        
        Args:
            hidden_states: [batch, seq, hidden_size]
        
        Returns:
            scores: [batch, seq, num_experts]
        """
        # [batch, seq, hidden] @ [hidden, num_experts] -> [batch, seq, num_experts]
        logits = torch.matmul(hidden_states, self.expert_centroids.T)
        return torch.sigmoid(logits)
    
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """路由 token 到专家
        
        Args:
            hidden_states: [batch, seq, hidden_size]
        
        Returns:
            topk_indices: [batch, seq, K_r] - 选中的专家索引
            gate_values: [batch, seq, K_r] - 归一化的 gate 值
        """
        # Eq. 15: 计算亲和度分数
        affinity_scores = self.compute_affinity_scores(hidden_states)
        
        # Eq. 16: 使用 bias 进行 Top-K 选择（bias 仅影响选择，不影响权重）
        routing_scores = affinity_scores + self.expert_bias
        
        # Top-K 选择
        _, topk_indices = torch.topk(
            routing_scores, 
            self.num_active_experts, 
            dim=-1
        )
        
        # 获取被选中专家的原始亲和度分数（不包含 bias）
        topk_scores = torch.gather(affinity_scores, -1, topk_indices)
        
        # Eq. 13, 14: 归一化 gate 值
        # g_{i,t} = s'_{i,t} / sum_j s'_{j,t}
        gate_values = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        return topk_indices, gate_values
    
    def update_bias(self, expert_load: torch.Tensor, step_size: float = 0.001):
        """动态更新 bias 以平衡负载
        
        训练时调用，若过载则减小 b_i，若欠载则增加 b_i
        
        Args:
            expert_load: [num_experts] - 每个专家处理的 token 数
            step_size: 更新步长
        """
        # 计算平均负载
        avg_load = expert_load.mean()
        
        # 相对负载差异
        load_diff = avg_load - expert_load
        
        # 更新 bias
        self.expert_bias.add_(load_diff * step_size)


class DeepSeekMoE(nn.Module):
    """
    DeepSeek MoE 层
    
    Eq. 12: h'_t = u_t + Σ FFN_i^{(s)}(u_t) + Σ g_{i,t} FFN_i^{(r)}(u_t)
    """
    
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.num_active_experts = config.num_active_experts
        
        # 共享专家 (Eq. 12 中的 FFN_i^{(s)})
        self.shared_experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, config.intermediate_size)
            for _ in range(self.num_shared_experts)
        ])
        
        # 路由专家 (Eq. 12 中的 FFN_i^{(r)})
        self.routed_experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, config.expert_intermediate_size)
            for _ in range(self.num_routed_experts)
        ])
        
        # 路由器
        self.router = ExpertRouter(config)
    
    def _compute_shared_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算共享专家输出
        
        Eq. 12 第一项: Σ_{i=1}^{N_s} FFN_i^{(s)}(u_t)
        """
        output = torch.zeros_like(hidden_states)
        for expert in self.shared_experts:
            output = output + expert(hidden_states)
        return output
    
    def _compute_routed_output(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        gate_values: torch.Tensor,
    ) -> torch.Tensor:
        """计算路由专家输出
        
        Eq. 12 第二项: Σ_{i=1}^{N_r} g_{i,t} FFN_i^{(r)}(u_t)
        
        实现方式：按专家分组处理以提高效率
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 展平以便处理
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]
        flat_indices = topk_indices.view(-1, self.num_active_experts)  # [batch*seq, K_r]
        flat_gates = gate_values.view(-1, self.num_active_experts)  # [batch*seq, K_r]
        
        # 输出缓冲区
        routed_output = torch.zeros_like(flat_hidden)
        
        # 逐专家计算
        for k in range(self.num_active_experts):
            expert_indices = flat_indices[:, k]  # [batch*seq]
            expert_gates = flat_gates[:, k:k+1]  # [batch*seq, 1]
            
            # 按专家 ID 分组处理
            for expert_id in range(self.num_routed_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = flat_hidden[mask]
                    expert_output = self.routed_experts[expert_id](expert_input)
                    
                    # 加权累加
                    weighted_output = expert_output * expert_gates[mask]
                    routed_output[mask] += weighted_output
        
        return routed_output.view(batch_size, seq_len, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] - u_t (Attention 输出)
        
        Returns:
            output: [batch, seq_len, hidden_size] - h'_t
        """
        # Step 1: 路由
        topk_indices, gate_values = self.router(hidden_states)
        
        # Step 2: 共享专家输出
        shared_output = self._compute_shared_output(hidden_states)
        
        # Step 3: 路由专家输出
        routed_output = self._compute_routed_output(
            hidden_states, topk_indices, gate_values
        )
        
        # Eq. 12: h'_t = u_t + shared_output + routed_output
        output = hidden_states + shared_output + routed_output
        
        return output
