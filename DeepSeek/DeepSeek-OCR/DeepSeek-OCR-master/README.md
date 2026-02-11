# DeepSeek-OCR Source Code

This folder contains the official source code for **DeepSeek-OCR** (first generation), an advanced document OCR model by DeepSeek.

## Structure

```
DeepSeek-OCR-master/
├── DeepSeek-OCR-hf/          # HuggingFace Transformers implementation
│   └── run_dpsk_ocr.py       # Simple inference script using AutoModel
└── DeepSeek-OCR-vllm/        # vLLM-based implementation for high-throughput serving
    ├── config.py             # Model configuration
    ├── deepencoder/          # Vision encoder modules
    ├── deepseek_ocr.py       # Main OCR model
    ├── process/              # Image preprocessing utilities
    ├── run_dpsk_ocr_eval_batch.py  # Batch evaluation
    ├── run_dpsk_ocr_image.py       # Single image inference
    └── run_dpsk_ocr_pdf.py         # PDF document processing
```

## Key Features

- **Model**: `deepseek-ai/DeepSeek-OCR` from HuggingFace Hub
- **Flash Attention 2**: Optimized attention for faster inference
- **Multi-Resolution Modes**:
  - Tiny: 512×512
  - Small: 640×640
  - Base: 1024×1024
  - Large: 1280×1280
  - Gundam: 1024 base with 640 crop (best for complex documents)

## Quick Start (HuggingFace)

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
res = model.infer(tokenizer, prompt=prompt, image_file='doc.jpg', 
                  base_size=1024, image_size=640, crop_mode=True)
```

## Source

Official DeepSeek-AI repository for document understanding and OCR capabilities.
