from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from util import index_exists

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = (
    "GPT2 is a model developed by OpenAI."
    if not index_exists(sys.argv, 1)
    else sys.argv[1]
)

# python gpt2.py "hi gpt"
print("prompt: ", prompt)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,
    max_length=100,
)

print("output: <output>", tokenizer.batch_decode(gen_tokens)[0], "</output>")
