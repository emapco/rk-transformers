---
{{ card_data }}
---

# {{ model_name }} (RKNN2)

> This is an RKNN-compatible version of the [{{ base_model }}]({{ base_model_url }}) model. It has been optimized for Rockchip NPUs using the [rk-transformers](https://github.com/emapco/rk-transformers) library.

## Model Details

- **Original Model:** [{{ base_model }}]({{ base_model_url }})
- **Target Platform:** {{ target_platform }}
- **rknn-toolkit2 Version:** {{ rknn_version }}
- **rk-transformers Version:** {{ rk_transformers_version }}

### Available Model Files

| Model File | Optimization Level | Quantization | File Size |
| :--------- | :----------------- | :----------- | :-------- |
{%- for model in available_models %}
| [{{ model.path }}](./{{ model.path }}) | {{ model.optimization_level }} | {{ model.quantization }} | {{ model.file_size }} |
{%- endfor %}

## Usage

### Installation

Install `rk-transformers` with inference dependencies to use this model:

```bash
pip install rk-transformers[inference]
```{% if is_sentence_transformer %}

#### Sentence Transformers

```python
from rktransformers import RKSentenceTransformer

model = RKSentenceTransformer(
    "{{ example_model_path }}",
    model_kwargs={
        "platform": "{{ target_platform }}",
        "core_mask": "auto",{% if example_file_name %}
        "file_name": "{{ example_file_name }}"{% endif %}
    }
)

sentences = ["This is a test sentence", "Another example"]
embeddings = model.encode(sentences)
print(embeddings.shape){% if optimized_model_path %}

# Load specific optimized/quantized model file
model = RKSentenceTransformer(
    "{{ example_model_path }}",
    model_kwargs={
        "platform": "{{ target_platform }}",
        "file_name": "{{ optimized_model_path }}"
    }
){% endif %}
```{% endif %}{% if usage_example %}

#### RK-Transformers API

{{ usage_example }}{% else %}

#### Unsupported RKNN Model

This model type is not currently supported by the RK-Transformers API. You will need to write custom inference code using `rknn-toolkit-lite2`.

```bash
pip install rknn-toolkit-lite2
```{% endif %}

## Configuration

The full configuration for all exported RKNN models is available in the [config.json](./config.json) file.
