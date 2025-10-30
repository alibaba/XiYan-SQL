
# download from modelscope
from modelscope import snapshot_download
# model_name = 'Qwen/Qwen2.5-Coder-3B-Instruct'
# model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
model_name = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
model_dir = snapshot_download(model_name, cache_dir='./model/')

# download from huggingface
# from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoTokenizer.from_pretrained(model_name, cache_dir='./model/')
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./model/')













