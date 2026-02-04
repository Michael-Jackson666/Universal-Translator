"""
DeepSeek-V3 模型架构实现 - 主入口

基于论文：DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437)

模块结构:
- config.py: 配置类
- layers.py: 基础组件 (RMSNorm, RoPE, ExpertFFN)
- mla.py: Multi-Head Latent Attention
- moe.py: DeepSeek MoE 层
- mtp.py: Multi-Token Prediction
- block.py: Transformer Block
- model.py: 完整模型
- deepseek_v3.py: 主入口和测试

总参数量: 671B
激活参数量: 37B (per token)
"""

import torch

# 导入所有模块
from config import DeepSeekV3Config, create_small_config
from layers import RMSNorm, RotaryEmbedding, ExpertFFN, apply_rotary_pos_emb
from mla import MultiHeadLatentAttention
from moe import DeepSeekMoE, ExpertRouter
from mtp import MultiTokenPrediction, MultiTokenPredictionModule
from block import DeepSeekV3Block
from model import DeepSeekV3Model, DeepSeekV3ForCausalLM


def create_small_model_for_test() -> DeepSeekV3ForCausalLM:
    """创建一个小型测试模型"""
    config = create_small_config()
    return DeepSeekV3ForCausalLM(config)


def test_individual_components():
    """测试各个组件"""
    print("\n" + "=" * 50)
    print("测试各个组件")
    print("=" * 50)
    
    config = create_small_config()
    batch_size = 2
    seq_len = 16
    
    # 测试输入
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # 1. 测试 RMSNorm
    print("\n1. RMSNorm...")
    norm = RMSNorm(config.hidden_size)
    out = norm(hidden_states)
    print(f"   输入: {hidden_states.shape} -> 输出: {out.shape}")
    
    # 2. 测试 MLA
    print("\n2. Multi-Head Latent Attention...")
    mla = MultiHeadLatentAttention(config, layer_idx=0)
    out = mla(hidden_states, position_ids)
    print(f"   输入: {hidden_states.shape} -> 输出: {out.shape}")
    
    # 3. 测试 ExpertFFN
    print("\n3. ExpertFFN...")
    ffn = ExpertFFN(config.hidden_size, config.intermediate_size)
    out = ffn(hidden_states)
    print(f"   输入: {hidden_states.shape} -> 输出: {out.shape}")
    
    # 4. 测试 Router
    print("\n4. ExpertRouter...")
    router = ExpertRouter(config)
    indices, gates = router(hidden_states)
    print(f"   输入: {hidden_states.shape}")
    print(f"   专家索引: {indices.shape}, Gate 值: {gates.shape}")
    
    # 5. 测试 MoE
    print("\n5. DeepSeekMoE...")
    moe = DeepSeekMoE(config)
    out = moe(hidden_states)
    print(f"   输入: {hidden_states.shape} -> 输出: {out.shape}")
    
    # 6. 测试 Block
    print("\n6. DeepSeekV3Block...")
    block = DeepSeekV3Block(config, layer_idx=0)
    out = block(hidden_states, position_ids)
    print(f"   输入: {hidden_states.shape} -> 输出: {out.shape}")
    
    print("\n✓ 所有组件测试通过！")


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 50)
    print("测试完整模型")
    print("=" * 50)
    
    # 创建小型测试模型
    model = create_small_model_for_test()
    
    # 统计参数量
    params = model.count_parameters()
    print(f"\n模型参数量:")
    print(f"  总计: {params['total'] / 1e6:.2f}M")
    print(f"  Embedding: {params['embedding'] / 1e6:.2f}M")
    print(f"  Attention: {params['attention'] / 1e6:.2f}M")
    print(f"  MoE: {params['moe'] / 1e6:.2f}M")
    print(f"  LM Head: {params['lm_head'] / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\n输入形状: {input_ids.shape}")
    
    # 无标签前向传播
    loss, logits, mtp_logits = model(input_ids)
    print(f"输出 logits 形状: {logits.shape}")
    print(f"MTP logits 数量: {len(mtp_logits) if mtp_logits else 0}")
    
    # 有标签前向传播 (训练模式)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    loss, logits, mtp_logits = model(input_ids, labels=labels)
    print(f"损失值: {loss.item():.4f}")
    
    # 测试生成
    print("\n测试生成...")
    generated = model.generate(input_ids[:1, :8], max_new_tokens=10)
    print(f"生成序列形状: {generated.shape}")
    
    print("\n✓ 完整模型测试通过！")


def main():
    """主函数"""
    print("=" * 60)
    print("DeepSeek-V3 模型架构测试")
    print("=" * 60)
    
    # 测试各个组件
    test_individual_components()
    
    # 测试完整模型
    test_full_model()
    
    print("\n" + "=" * 60)
    print("所有测试完成！模型架构验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()

