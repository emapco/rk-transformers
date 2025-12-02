```python
from rktransformers import RKModelForCausalLM
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("{{ tokenizer_path }}")
model = RKModelForCausalLM.from_pretrained(
    "{{ example_model_path }}",
    core_mask="auto",{% if example_file_name %}
    file_name="{{ example_file_name }}"{% endif %}
)

inputs = tokenizer("My name is Arthur and I live in", return_tensors="np")
gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
output = tokenizer.batch_decode(gen_tokens)
print(output.shape){% if optimized_model_path %}

# Load specific optimized/quantized model file
model = RKModelForCausalLM.from_pretrained(
    "{{ example_model_path }}",
    file_name="{{ optimized_model_path }}"
){% endif %}
```
