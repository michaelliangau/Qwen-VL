from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM

torch.manual_seed(1234)
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("/home/michael/tmp/Qwen-VL/output_qwen/checkpoint-6250", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("/home/michael/tmp/Qwen-VL/output_qwen/checkpoint-6250", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("/home/michael/tmp/Qwen-VL/output_qwen/checkpoint-6250", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoPeftModelForCausalLM.from_pretrained(
    "/home/michael/Qwen-VL/output_qwen/checkpoint-6250", # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': '/home/michael/ai/projects/computer_agent/data/2_dot_qwen_eval/1.png'}, # Either a local path or an url
    {'text': 'Click on the green square.'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
