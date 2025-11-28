---
base_model: /home/manny/rk-transformers/tests/data/random_roberta
library_name: rk-transformers
model_name: random_roberta
tags:
- rknn
- rockchip
- npu
- rk-transformers
- rk3588
---

# random_roberta (RKNN2)

> This is an RKNN-compatible version of the [/home/manny/rk-transformers/tests/data/random_roberta](/home/manny/rk-transformers/tests/data/random_roberta) model. It has been optimized for Rockchip NPUs using the [rk-transformers](https://github.com/emapco/rk-transformers) library.

## Model Details

- **Original Model:** [/home/manny/rk-transformers/tests/data/random_roberta](/home/manny/rk-transformers/tests/data/random_roberta)
- **Target Platform:** rk3588
- **rknn-toolkit2 Version:** 2.3.2
- **rk-transformers Version:** 0.2.0

### Available Model Files

| Model File | Optimization Level | Quantization | File Size |
| :--------- | :----------------- | :----------- | :-------- |
| [model_b1_s16.rknn](./model_b1_s16.rknn) | 0 | float16 | 469.5 KB |
| [model_b1_s32.rknn](./model_b1_s32.rknn) | 0 | float16 | 467.2 KB |
| [model_b2_s32.rknn](./model_b2_s32.rknn) | 0 | float16 | 725.7 KB |
| [model_b4_s64.rknn](./model_b4_s64.rknn) | 0 | float16 | 1.1 MB |

## Usage

### Installation

Install `rk-transformers` to use this model:

```bash
pip install rk-transformers
```

#### RKTransformers API

```python
from rktransformers import RKRTModelForFeatureExtraction
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("random_roberta")
model = RKRTModelForFeatureExtraction.from_pretrained(
    "random_roberta",
    platform="rk3588",
    core_mask="auto",
    file_name="model_b1_s16.rknn"
)

inputs = tokenizer(
    ["Sample text for encoding"],
    padding="max_length",
    max_length=16,
    truncation=True,
    return_tensors="np"
)

outputs = model(**inputs)
print(outputs.shape)
```

## Configuration

The full configuration for all exported RKNN models is available in the [rknn.json](./rknn.json) file.