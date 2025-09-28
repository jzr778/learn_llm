import torch
import torch.nn as nn
import bitsandbytes as bnb 
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1", 
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
for i, param in enumerate(model.parameters()):
    param.requires_grad = False
    if param.ndim == 1: #把一维的小参数（bias、LayerNorm 等）单独升到 float32，保证数值稳定，其余权重保持原精度不动
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32) #将输出参数转为float32，权重参数不变
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(mode):
    trainable_params = 0
    all_param = 0
    for _,param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"训练参数: {trainable_params} ||所有参数: {all_param} || 训练参数占比%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

print_trainable_parameters(model)

import transformers
from datasets import load_dataset
dataset = load_dataset("Abirate/english_quotes")

def merge(row):
    row['prediction'] = row['quote'] + ' ->: ' + str(row['tags'])
    return row
dataset['train'] = dataset['train'].map(merge)

train_dataset = dataset.map(lambda x:tokenizer(x['prediction']), batched=True, remove_columns=['quote', 'author', 'tags', 'prediction'])

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
trainer = Trainer(
    model=model, 
    train_dataset=train_dataset['train'],
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="bloom-7b1-lora-0927"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False 
trainer.train()
