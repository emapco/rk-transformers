```python
import numpy as np
from rktransformers import RKModelForMultipleChoice
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{ tokenizer_path }}")
model = RKModelForMultipleChoice.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    core_mask="auto",{% if example_file_name %}
    file_name="{{ example_file_name }}"{% endif %}
)

prompt = "In Italy, pizza is served in slices."
choice0 = "It is eaten with a fork and knife."
choice1 = "It is eaten while held in the hand."
choice2 = "It is blended into a smoothie."
choice3 = "It is folded into a taco."

encoding = tokenizer(
    [prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3], return_tensors="np", padding=True
)
inputs = {k: np.expand_dims(v, 0) for k, v in encoding.items()}

outputs = model(**inputs)
logits = outputs.logits
print(logits.shape){% if optimized_model_path %}

# Load specific optimized/quantized model file
model = RKModelForMultipleChoice.from_pretrained(
    "{{ example_model_path }}",
    platform="{{ target_platform }}",
    file_name="{{ optimized_model_path }}"
){% endif %}
```
