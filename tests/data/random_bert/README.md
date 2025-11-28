---
base_model: /home/manny/rk-transformers/tests/data/random_bert
library_name: rk-transformers
model_name: random_bert
tags:
- rknn
- rockchip
- npu
- rk-transformers
- rk3588
---

# random_bert (RKNN2)

> This is an RKNN-compatible version of the [/home/manny/rk-transformers/tests/data/random_bert](/home/manny/rk-transformers/tests/data/random_bert) model. It has been optimized for Rockchip NPUs using the [rk-transformers](https://github.com/emapco/rk-transformers) library.

## Model Details

- **Original Model:** [/home/manny/rk-transformers/tests/data/random_bert](/home/manny/rk-transformers/tests/data/random_bert)
- **Target Platform:** rk3588
- **rknn-toolkit2 Version:** 2.3.2
- **rk-transformers Version:** 0.3.0

### Available Model Files

| Model File | Optimization Level | Quantization | File Size |
| :--------- | :----------------- | :----------- | :-------- |
| [rknn/model_b1_s32_o3.rknn](./rknn/model_b1_s32_o3.rknn) | 3 | float16 | 2.2 MB |

## Usage

### Installation

Install `rk-transformers` with inference dependencies to use this model:

```bash
pip install rk-transformers[inference]
```

#### RK-Transformers API

```python
from rktransformers import RKModelForFeatureExtraction
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("random_bert")
model = RKModelForFeatureExtraction.from_pretrained(
    "random_bert",
    platform="rk3588",
    core_mask="auto",
    file_name="rknn/model_b1_s32_o3.rknn"
)

inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)

# Load specific optimized/quantized model file
model = RKModelForFeatureExtraction.from_pretrained(
    "random_bert",
    platform="rk3588",
    file_name="rknn/model_b1_s32_o3.rknn"
)
```

## Configuration

The full configuration for all exported RKNN models is available in the [config.json](./config.json) file.