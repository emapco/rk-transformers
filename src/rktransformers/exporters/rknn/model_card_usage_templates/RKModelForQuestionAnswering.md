```python
from rktransformers import RKModelForQuestionAnswering
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{ tokenizer_path }}")
model = RKModelForQuestionAnswering.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    core_mask="auto",{% if example_file_name %}
    file_name="{{ example_file_name }}"{% endif %}
)

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="np")
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape)
print(end_logits.shape){% if optimized_model_path %}

# Load specific optimized/quantized model file
model = RKModelForQuestionAnswering.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    file_name="{{ optimized_model_path }}"
){% endif %}
```
