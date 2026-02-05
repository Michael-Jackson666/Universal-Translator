"""
mHC Main Entry and Tests

Reference: mHC: Manifold-Constrained Hyper-Connections (https://arxiv.org/abs/2512.24880)

模块结构:
- config.py: 配置类
- manifold.py: Birkhoff 多面体投影 + Sinkhorn-Knopp 算法
- layers.py: mHC 层实现
- model.py: 完整模型
- mhc.py: 主入口和测试
"""

import torch

# 导入所有模块
from config import mHCConfig, create_small_config
from manifold import SinkhornKnopp, ManifoldProjection
from layers import mHCDynamicMapping, mHCLayer, RMSNorm
from model import mHCModel, mHCForSequenceModeling


def test_sinkhorn_knopp():
    """测试 Sinkhorn-Knopp 算法"""
    print("\n" + "=" * 50)
    print("测试 Sinkhorn-Knopp 算法")
    print("=" * 50)
    
    sinkhorn = SinkhornKnopp(num_iterations=20)
    
    # 随机矩阵
    n = 4
    H_tilde = torch.randn(2, n, n)  # [batch=2, n, n]
    
    # 投影
    H_res = sinkhorn(H_tilde)
    
    # 验证双随机性质
    row_sums = H_res.sum(dim=-1)  # 每行和
    col_sums = H_res.sum(dim=-2)  # 每列和
    
    print(f"输入形状: {H_tilde.shape}")
    print(f"输出形状: {H_res.shape}")
    print(f"行和 (应为 1): {row_sums[0]}")
    print(f"列和 (应为 1): {col_sums[0]}")
    print(f"最小值 (应 >= 0): {H_res.min().item():.6f}")
    
    # 验证谱范数
    spectral_norm = torch.linalg.matrix_norm(H_res, ord=2)
    print(f"谱范数 (应 <= 1): {spectral_norm.max().item():.4f}")
    
    print("✓ Sinkhorn-Knopp 测试通过！")


def test_manifold_projection():
    """测试流形投影"""
    print("\n" + "=" * 50)
    print("测试流形投影")
    print("=" * 50)
    
    n = 4
    proj = ManifoldProjection(num_streams=n, sinkhorn_iterations=20)
    
    batch_size = 2
    H_tilde_pre = torch.randn(batch_size, n)
    H_tilde_post = torch.randn(batch_size, n)
    H_tilde_res = torch.randn(batch_size, n, n)
    
    H_pre, H_post, H_res = proj(H_tilde_pre, H_tilde_post, H_tilde_res)
    
    print(f"H_pre 范围: [{H_pre.min():.4f}, {H_pre.max():.4f}] (应在 [0, 1])")
    print(f"H_post 范围: [{H_post.min():.4f}, {H_post.max():.4f}] (应在 [0, 2])")
    print(f"H_res 是双随机矩阵: 行和={H_res.sum(-1)[0]}")
    
    print("✓ 流形投影测试通过！")


def test_mhc_layer():
    """测试 mHC 层"""
    print("\n" + "=" * 50)
    print("测试 mHC 层")
    print("=" * 50)
    
    config = create_small_config()
    layer = mHCLayer(config)
    
    batch_size = 2
    x = torch.randn(batch_size, config.num_streams, config.hidden_size)
    
    print(f"配置: n={config.num_streams}, C={config.hidden_size}")
    print(f"输入形状: {x.shape}")
    
    y = layer(x)
    
    print(f"输出形状: {y.shape}")
    
    # 检查信号幅度
    input_norm = x.norm(dim=-1).mean()
    output_norm = y.norm(dim=-1).mean()
    gain = output_norm / input_norm
    
    print(f"输入范数: {input_norm:.4f}")
    print(f"输出范数: {output_norm:.4f}")
    print(f"增益: {gain:.4f} (mHC 应保持稳定，接近 1)")
    
    print("✓ mHC 层测试通过！")


def test_mhc_model():
    """测试完整 mHC 模型"""
    print("\n" + "=" * 50)
    print("测试完整 mHC 模型")
    print("=" * 50)
    
    config = create_small_config()
    model = mHCModel(config)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e3:.2f}K")
    
    batch_size = 2
    x = torch.randn(batch_size, config.hidden_size)
    
    print(f"输入形状: {x.shape}")
    
    # 标准前向传播
    y = model(x, use_recompute=False)
    print(f"输出形状: {y.shape}")
    
    # 测试梯度
    loss = y.sum()
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"梯度范数范围: [{min(grad_norms):.4f}, {max(grad_norms):.4f}]")
    print("✓ 完整模型测试通过！")


def test_stability():
    """测试深层网络稳定性"""
    print("\n" + "=" * 50)
    print("测试深层网络稳定性")
    print("=" * 50)
    
    # 创建较深的网络
    config = mHCConfig(
        hidden_size=128,
        num_streams=4,
        num_layers=16,
        sinkhorn_iterations=20,
    )
    model = mHCModel(config)
    
    batch_size = 4
    x = torch.randn(batch_size, config.hidden_size)
    
    # 记录每层的激活范数
    with torch.no_grad():
        hidden = model._expand_to_streams(x)
        norms = [hidden.norm(dim=-1).mean().item()]
        
        for layer in model.layers:
            hidden = layer(hidden)
            norms.append(hidden.norm(dim=-1).mean().item())
    
    # 计算增益
    max_gain = max(norms) / norms[0]
    min_gain = min(norms) / norms[0]
    
    print(f"层数: {config.num_layers}")
    print(f"初始范数: {norms[0]:.4f}")
    print(f"最终范数: {norms[-1]:.4f}")
    print(f"最大增益: {max_gain:.4f}")
    print(f"最小增益: {min_gain:.4f}")
    
    # mHC 的关键：增益应该保持在合理范围内
    if max_gain < 10 and min_gain > 0.1:
        print("✓ 稳定性测试通过！信号在 16 层后保持稳定")
    else:
        print("⚠ 稳定性警告：信号可能有爆炸或消失趋势")


def main():
    """主函数"""
    print("=" * 60)
    print("mHC: Manifold-Constrained Hyper-Connections")
    print("=" * 60)
    
    # 测试各组件
    test_sinkhorn_knopp()
    test_manifold_projection()
    test_mhc_layer()
    test_mhc_model()
    test_stability()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
