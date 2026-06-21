seed = 0

base_model_path = "/kaggle/input/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507"
pretrain_lora_path = None
lora_path = "/kaggle/working/pseudo_lora"
use_gptq = "gptq" in base_model_path

positive = "Yes"
negative = "No"
judge_words = "Violation:"
system_prompt = '''You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule in this subreddit. Only respond Yes/No.'''
frac = 0.05
use_train = True

# 混合损失函数权重
main_loss_weight = 0.8
aux_loss_weight = 0.2

import kagglehub

deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)