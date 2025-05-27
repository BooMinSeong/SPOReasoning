from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
model = AutoModelForCausalLM.from_pretrained("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")