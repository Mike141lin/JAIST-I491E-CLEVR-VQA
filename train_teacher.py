import os
import torch
import pandas as pd
import ast
import random
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./output/teacher_3b"
DATA_ROOT = "custom_dataset"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")

class TeacherDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['file']
        img_path = os.path.join(self.img_dir, img_name)
        try: image = Image.open(img_path).convert("RGB")
        except: image = Image.new('RGB', (224, 224), 'black')
        raw_exp = row['explanation']
        try:
            exp_list = ast.literal_eval(raw_exp)
            explanation = random.choice(exp_list) if isinstance(exp_list, list) else str(raw_exp)
        except: explanation = str(raw_exp)
        
        # 简单格式 Answer First
        target_text = f"Answer: {row['answer']}\nExplanation: {explanation}"
        
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{row['question']}\nProvide the answer and a detailed explanation."}]}, {"role": "assistant", "content": [{"type": "text", "text": target_text}]}]
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=process_vision_info(conversation)[0], padding=False, return_tensors="pt")
        return {"input_ids": inputs["input_ids"][0], "labels": inputs["input_ids"][0], "pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"], "attention_mask": inputs["attention_mask"][0]}

def collate_fn(examples):
    input_ids = torch.nn.utils.rnn.pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=151643)
    labels = torch.nn.utils.rnn.pad_sequence([e["labels"] for e in examples], batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence([e["attention_mask"] for e in examples], batch_first=True, padding_value=0)
    pixel_values = torch.cat([e["pixel_values"] for e in examples], dim=0)
    image_grid_thw = torch.cat([e["image_grid_thw"] for e in examples], dim=0)
    return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "attention_mask": attention_mask}

if __name__ == "__main__":
    print("🧠 Training Teacher (Qwen2.5-3B) with Gradient Checkpointing...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
    
    # === 关键修复: 启用输入梯度 ===
    # 这行代码是 Gradient Checkpointing + LoRA 必须的！
    model.enable_input_require_grads()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    peft_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type=TaskType.CAUSAL_LM, lora_dropout=0.05)
    model = get_peft_model(model, peft_config)
    
    dataset = TeacherDataset(TRAIN_CSV, TRAIN_IMG_DIR, processor)
    
    # 显存优化配置: Batch=1, Accum=16, Checkpointing=True
    args = TrainingArguments(output_dir=OUTPUT_DIR, per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=2e-4, num_train_epochs=3, bf16=True, gradient_checkpointing=True, logging_steps=10, save_strategy="epoch", report_to="none", remove_unused_columns=False)
    
    Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate_fn).train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    print("✅ Teacher Done!")
