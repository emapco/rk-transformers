```python
from rktransformers import RKModelForFeatureExtraction
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{ tokenizer_path }}")
model = RKModelForFeatureExtraction.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    core_mask="auto",{% if example_file_name %}
    file_name="{{ example_file_name }}"{% endif %}
)

inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape){% if optimized_model_path %}

# Load specific optimized/quantized model file
model = RKModelForFeatureExtraction.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    file_name="{{ optimized_model_path }}"
){% endif %}
```
