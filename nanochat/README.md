# NanoChat

从零开始实现的极简大语言模型（LLM）集合，用最少的代码展示核心算法。

## 项目目标

- 🎯 **教育目的**：用最简洁的代码揭示 LLM 的核心原理
- 🔬 **无依赖**：纯 Python 实现，不依赖 PyTorch/TensorFlow
- 📚 **循序渐进**：从最简单的实现到带详细注释的版本

## 目录结构

```
nanochat/
├── README.md           # 本文件
└── microgpt/           # 最小化 GPT 实现
    ├── README.md       # MicroGPT 详细说明
    ├── microgpt.py     # 原始简洁版（~200行）
    └── microgpt-detail.py  # 详细注释版（含数学推导）
```

## 核心概念

这个项目展示了构建 LLM 所需的全部核心组件：

| 组件 | 作用 | 代码位置 |
|------|------|----------|
| **Tokenizer** | 文本 ↔ 数字转换 | 字符级分词 |
| **Autograd** | 自动微分引擎 | `Value` 类 |
| **Transformer** | 注意力 + MLP | `gpt()` 函数 |
| **Adam** | 优化器 | 训练循环中 |
| **Inference** | 文本生成 | Temperature 采样 |

## 学习路径

1. **入门**：先阅读 `microgpt/microgpt.py`，理解整体结构（~200行）
2. **深入**：阅读 `microgpt/microgpt-detail.py`，理解每个运算的数学原理
3. **实践**：修改超参数，观察训练效果变化

## 快速开始

```bash
cd nanochat/microgpt
python microgpt.py
```

程序会自动下载人名数据集，训练一个字符级语言模型，然后生成新的"人名"。

## 致谢

- [Andrej Karpathy](https://github.com/karpathy) - MicroGPT 原始实现
- [makemore](https://github.com/karpathy/makemore) - 人名数据集

## 相关资源

- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Karpathy 的 GPT 教程视频
- [micrograd](https://github.com/karpathy/micrograd) - 极简自动微分引擎
- [nanoGPT](https://github.com/karpathy/nanoGPT) - 简洁的 GPT-2 训练代码
