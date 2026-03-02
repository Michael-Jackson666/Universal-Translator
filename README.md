# Zero2Hero-AI 🚀

**From first principles to state-of-the-art: A hands-on journey implementing LLMs, decoding DeepSeek, and building real-world AI.**

> 从第一性原理到前沿技术：亲手实现大语言模型，深度解读 DeepSeek，构建真实世界的 AI 应用。

---

## 📖 项目简介

这是一个记录我从零学习 AI 全过程的项目仓库。从基础原理到前沿论文，从代码实现到实际应用，涵盖了大语言模型、推理系统、机器翻译等多个核心领域。

### 🎯 核心内容

| 模块 | 描述 |
|------|------|
| **llm-scratch** | 📚 从零构建大语言模型 (GPT)，包含文本处理、注意力机制、模型实现、预训练、微调全流程 |
| **nanochat** | ⚡ 极简 LLM 集合，纯 Python 无框架依赖，用最少代码揭示 LLM 核心原理 |
| **DeepSeek** | 🔬 DeepSeek 系列论文精读与复现，包括 MoE、Engram、mHC 等核心技术解析 |
| **reasoning-scratch** | 🧠 从零构建推理模型，探索逻辑推理与搜索算法 |
| **Eng2Fren** | 🌍 基于 Transformer 的英法翻译模型，从训练到部署的完整实践 |
| **huggingface** | 🤗 Hugging Face 使用教程与实战案例 |
| **ai_test** | 💼 AI 算法学习、Transformer 教程、大厂机试真题 |

---

## 📁 项目结构

```
Zero2Hero-AI/
├── llm-scratch/           # 🔥 从零构建大语言模型
│   ├── chap2-work_with_text_data/  # 文本数据处理与分词
│   ├── chap3-attention_mechanisms/ # 注意力机制详解
│   ├── chap4-implement_gpt_model/  # GPT 模型完整实现
│   ├── chap5-pretraining/          # 预训练技术
│   ├── chap6-fine-tuning-for-classification/  # 分类任务微调
│   ├── chap7-fine-tuning-to-follow-instruction/  # 指令微调 (SFT)
│   ├── appendix-A/                 # 分布式训练 (DDP)
│   ├── appendix-D/                 # 进阶训练技巧
│   └── appendix-E/                 # LoRA 微调
│
├── nanochat/              # ⚡ 极简 LLM 实现（纯 Python）
│   └── microgpt/                   # 最小化 GPT，含自动微分引擎
│
├── DeepSeek/              # 🔬 DeepSeek 论文精读与复现
│   ├── DeepSeek-MoE/               # MoE 混合专家模型解析
│   ├── Engram/                     # Engram 技术详解
│   ├── mHC/                        # mHC 架构分析
│   └── DeepThink.md                # 深度思考笔记
│
├── reasoning-scratch/     # 🧠 从零构建推理模型
│   └── README.md                   # 逻辑推理、搜索算法
│
├── Eng2Fren/              # 🌍 英法翻译模型
│   ├── transformer.py              # Transformer 模型实现
│   ├── transformer-d2l.py          # 训练脚本
│   ├── simple_translator.py        # 交互式翻译器
│   └── batch_translate.py          # 批量翻译工具
│
├── huggingface/           # 🤗 Hugging Face 教程
│   ├── demo.ipynb                  # 实战 Demo
│   └── huggingface_examples.py     # 使用示例
│
└── ai_test/               # 💼 AI 算法学习与面试
    ├── Transformer/                # Transformer 完整教程
    ├── ACM/                        # ACM 算法练习
    ├── Whiteboard_Coding/          # 白板编程题
    └── 2025/                       # 大厂机试真题 (2025)
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/Michael-Jackson666/Zero2Hero-AI.git
cd Zero2Hero-AI

# 安装依赖
pip install -r requirements.txt
```

### 学习路线推荐

```
1️⃣ 基础入门
   └── ai_test/Transformer/tutorial.ipynb  # 理解 Transformer 架构

2️⃣ 极简实现
   └── nanochat/microgpt/  # 纯 Python，200 行理解 LLM 全貌

3️⃣ 动手实践
   └── Eng2Fren/  # 构建你的第一个翻译模型

4️⃣ 深入原理
   └── llm-scratch/  # 从零实现 GPT，逐章深入

5️⃣ 前沿探索
   └── DeepSeek/  # 精读最新论文，理解 MoE 等前沿技术

6️⃣ 工程应用
   └── huggingface/  # 学习工业级工具链
```

---

## 📚 各模块详解

### ⚡ nanochat - 极简 LLM 实现

用最少的代码揭示 LLM 的核心原理，纯 Python 实现，无需 PyTorch/TensorFlow：

- **microgpt.py**: ~200 行实现完整 GPT，含字符级分词、自动微分、Transformer、Adam 优化器
- **microgpt-detail.py**: 带详细数学推导注释的版本，适合深入理解每个运算

```bash
# 训练字符级语言模型并生成文本
cd nanochat/microgpt
python microgpt.py
```

---

### 🔥 llm-scratch - 从零构建大语言模型

跟随《Build a Large Language Model From Scratch》一书，逐章实现 GPT 模型：

- **Chapter 2**: BPE 分词、词嵌入、位置编码
- **Chapter 3**: 自注意力、多头注意力、因果掩码
- **Chapter 4**: GPT 架构、LayerNorm、前馈网络
- **Chapter 5**: 预训练、交叉熵损失、文本生成
- **Chapter 6**: 分类任务微调
- **Chapter 7**: 指令微调 (SFT)
- **Appendix**: DDP 分布式训练、LoRA 高效微调

### 🔬 DeepSeek - 论文精读与复现

深度解读 DeepSeek 系列技术：

- **DeepSeek-MoE**: 混合专家模型架构，理解稀疏激活与负载均衡
- **Engram**: 记忆增强技术
- **mHC**: 多头因果架构

### 🌍 Eng2Fren - 英法翻译模型

完整的 Seq2Seq 翻译项目：

```bash
# 训练模型
python Eng2Fren/transformer-d2l.py

# 交互式翻译
python Eng2Fren/simple_translator.py

# 批量翻译
python Eng2Fren/batch_translate.py
```

### 🤗 huggingface - 实战教程

学习使用 Hugging Face 生态：

- Transformers 库使用
- 预训练模型加载与推理
- 自定义模型微调

---

## 🎯 学习目标

- ✅ 理解 Transformer 架构的每一个细节
- ✅ 能够从零实现一个完整的 GPT 模型
- ✅ 掌握预训练、微调的完整流程
- ✅ 理解 DeepSeek 等前沿模型的核心创新
- ✅ 能够使用 Hugging Face 进行实际开发
- ✅ 具备 AI 岗位面试的算法能力

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！

---

## 📄 许可证

MIT License