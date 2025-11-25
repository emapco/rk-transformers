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

Install `rk-transformers` to use this model:

```bash
pip install rk-transformers
```
{% if is_sentence_transformer %}
#### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from rktransformers import patch_sentence_transformer

# Apply RKNN backend patch
patch_sentence_transformer()

# Load model with RKNN backend
model = SentenceTransformer(
    "{{ example_model_path }}",
    backend="rknn",
    model_kwargs={
        "platform": "{{ target_platform }}",
        "core_mask": "auto",{% if example_file_name %}
        "file_name": "{{ example_file_name }}"{% endif %}
    }
)

# Generate embeddings
sentences = ["This is a test sentence", "Another example"]
embeddings = model.encode(sentences)
print(embeddings.shape)
{% if optimized_model_path %}
# Load specific optimized/quantized model file
model = SentenceTransformer(
    "{{ example_model_path }}",
    backend="rknn",
    model_kwargs={
        "platform": "{{ target_platform }}",
        "file_name": "{{ optimized_model_path }}"
    }
){% endif %}
```{% endif %}
#### RKTransformers API

```python
from rktransformers import {{ rk_model_class }}
from transformers import AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("{{ tokenizer_path }}")
model = {{ rk_model_class }}.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    core_mask="auto",{% if example_file_name %}
    file_name="{{ example_file_name }}"{% endif %}
)

# Tokenize and run inference
inputs = tokenizer(
    ["Sample text for encoding"],
    padding="max_length",
    max_length={{ max_seq_length }},
    truncation=True,
    return_tensors="np"
)

outputs = model(**inputs)
print(outputs.shape)
{% if optimized_model_path %}
# Load specific optimized/quantized model file
model = {{ rk_model_class }}.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    file_name="{{ optimized_model_path }}"
){% endif %}
```

## Configuration

The full configuration for all exported RKNN models is available in the [rknn.json](./rknn.json) file.
