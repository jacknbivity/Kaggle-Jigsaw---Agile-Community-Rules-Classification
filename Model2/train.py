import torch
import torch.nn as nn
import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_torch_bf16_gpu_available

from utils import *
from constants import *

# 自定义分类头
class AuxiliaryClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hidden_states):
        return self.classifier(hidden_states)

# 自定义Trainer实现混合损失
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, num_aux_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_aux_classes = num_aux_classes
        self.aux_head = None
        
        if num_aux_classes is not None:
            # 获取模型的隐藏层大小
            hidden_size = self.model.config.hidden_size
            self.aux_head = AuxiliaryClassificationHead(hidden_size, num_aux_classes)
            self.aux_head = self.aux_head.to(self.model.device)
            
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 从inputs中提取aux_label
        aux_labels = inputs.pop("aux_label", None)
        
        # 计算主损失（原始的SFT损失）
        if return_outputs:
            main_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
            main_loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
            outputs = None
        
        # 如果有辅助标签且定义了辅助分类头，计算辅助损失
        if aux_labels is not None and self.aux_head is not None:
            # 重新运行forward来获取hidden states
            with torch.no_grad() if not model.training else torch.enable_grad():
                model_outputs = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # 获取最后一层的hidden states，取最后一个token的表示
            last_hidden_state = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # 使用最后一个token的hidden state进行分类
            # 找到每个序列中最后一个非padding token的位置
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                pooled_hidden = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            else:
                pooled_hidden = last_hidden_state[:, -1, :]
            
            # 通过辅助分类头得到logits
            aux_logits = self.aux_head(pooled_hidden)
            
            # 计算辅助损失（交叉熵）
            aux_loss_fct = nn.CrossEntropyLoss()
            aux_loss = aux_loss_fct(aux_logits, aux_labels.to(model.device))
            
            # 混合损失
            combined_loss = main_loss_weight * main_loss + aux_loss_weight * aux_loss
            
            if return_outputs:
                return (combined_loss, outputs)
            return combined_loss
        
        # 如果没有辅助标签，只返回主损失
        if return_outputs:
            return (main_loss, outputs)
        return main_loss

def main():
    # 获取数据集和规则信息
    df, rule_to_id, num_rules = get_df()
    train_dataset = build_dataset(df)
    num_aux_classes = num_rules + 1  # n个违规类型 + 1个不违规类型
    
    print(f"Auxiliary classification head will have {num_aux_classes} classes")
    
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
        trainer = CustomSFTTrainer(
            num_aux_classes=num_aux_classes,
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )
        trainer.train()
        trainer.save_model(lora_path)
        # 同时保存辅助分类头
        if trainer.aux_head is not None:
            torch.save(trainer.aux_head.state_dict(), f"{lora_path}/aux_head.pt")
            print(f"Auxiliary classification head saved to {lora_path}/aux_head.pt")
    else:
        peft_model = get_peft_model(model, lora_config)
        peft_model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

if __name__ == "__main__":
    main()