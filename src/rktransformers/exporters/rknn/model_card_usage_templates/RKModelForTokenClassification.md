```python
from rktransformers import RKModelForTokenClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{ tokenizer_path }}")
model = RKModelForTokenClassification.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    core_mask="auto",{% if example_file_name %}
    file_name="{{ example_file_name }}"{% endif %}
)

inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")
outputs = model(**inputs)
logits = outputs.logits
print(logits.shape){% if optimized_model_path %}

# Load specific optimized/quantized model file
model = RKModelForTokenClassification.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    file_name="{{ optimized_model_path }}"
){% endif %}
```
