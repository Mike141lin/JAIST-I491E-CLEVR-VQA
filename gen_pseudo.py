import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_ADAPTER = "./output/teacher_3b/final_adapter" 
DATA_ROOT = "custom_dataset"
UNLABELED_CSV = os.path.join(DATA_ROOT, "train_non_labels.csv")
UNLABELED_IMG_DIR = os.path.join(DATA_ROOT, "train_non_labels")
OUTPUT_PSEUDO_CSV = os.path.join(DATA_ROOT, "train_pseudo_labels_3b.csv")

def generate():
    print("🔮 Generating Pseudo Labels (Qwen2.5-3B)...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
    model = PeftModel.from_pretrained(base, TEACHER_ADAPTER).eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    
    df = pd.read_csv(UNLABELED_CSV)
    valid_data = []
    
    # 使用 Self-Consistency 清洗数据 (保留 0.913 版本的高效逻辑)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['file']
        img_path = os.path.join(UNLABELED_IMG_DIR, img_name)
        try: image = Image.open(img_path).convert("RGB")
        except: continue
        
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{row['question']}\nProvide the answer and a detailed explanation."}]}]
        text_in = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_in], images=process_vision_info(conversation)[0], padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, num_return_sequences=3)
            
        candidates = []
        for i in range(3):
            out = processor.decode(generated_ids[i][len(inputs.input_ids[0]):], skip_special_tokens=True)
            try:
                if "Answer:" in out:
                    ans = out.split("Answer:")[1].split("Explanation:")[0].strip().lower()
                    candidates.append((ans, out))
            except: continue
            
        if not candidates: continue
        answers = [c[0] for c in candidates]
        most_common_ans, count = Counter(answers).most_common(1)[0]
        
        if count >= 2 and most_common_ans != "":
            best_text = next(c[1] for c in candidates if c[0] == most_common_ans)
            try:
                parts = best_text.split("Explanation:")
                final_ans = parts[0].replace("Answer:", "").strip()
                final_exp = parts[1].strip()
                
                valid_data.append({
                    "id": row['id'], 
                    "file": img_name, 
                    "question": row['question'],
                    "answer": final_ans, 
                    "explanation": f"['{final_exp}']", 
                    "source_folder": "train_non_labels"
                })
            except: pass

    pd.DataFrame(valid_data).to_csv(OUTPUT_PSEUDO_CSV, index=False)
    print(f"🎉 Pseudo Labels Generated: {len(valid_data)}")

if __name__ == "__main__":
    generate()
