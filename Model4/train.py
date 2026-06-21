import torch
import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_torch_bf16_gpu_available

from utils import *
from constants import *

def main():
    train_dataset = build_dataset(get_df())
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    training_args = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=False,  # 强制使用 FP16，与推理保持一致
        fp16=True,   # 使用 torch.float16
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )

    if use_gptq:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,     
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if pretrain_lora_path:
        model = PeftModel.from_pretrained(model, pretrain_lora_path)
        model = model.merge_and_unload()

    if len(train_dataset) > 0:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )
        trainer.train()
        trainer.save_model(lora_path)
    else:
        peft_model = get_peft_model(model, lora_config)
        peft_model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

if __name__ == "__main__":
    main()