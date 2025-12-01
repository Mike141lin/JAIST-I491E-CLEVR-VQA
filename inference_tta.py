import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# === 关键修正: 指向存在的文件夹 student_3b ===
ADAPTER_PATH = "./output/student_3b/final_adapter"
OUTPUT_FILE = "submission_tta.csv"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

DATA_ROOT = "custom_dataset"
TEST_CSV = os.path.join(DATA_ROOT, "test_non_labels.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")

print(f"🚀 Running Multi-Scale TTA (1.0x, 1.2x, 0.85x) | Model: {ADAPTER_PATH}")

def run_inference():
    # 双重检查路径
    if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        print(f"❌ 严重错误: 依然找不到模型文件: {ADAPTER_PATH}")
        return

    print("Loading model...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    
    df = pd.read_csv(TEST_CSV)
    results = []
    
    print(f"Processing {len(df)} images with TTA...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['file']
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        try: 
            raw_image = Image.open(img_path).convert("RGB")
        except:
            raw_image = Image.new('RGB', (224, 224), 'black')
            
        # === TTA: 准备 3 张不同大小的图 ===
        w, h = raw_image.size
        images_tta = [
            raw_image,                                      # 1. 原图
            raw_image.resize((int(w*1.2), int(h*1.2))),     # 2. 放大 (看细节)
            raw_image.resize((int(w*0.85), int(h*0.85)))    # 3. 缩小 (看全局)
        ]
        
        candidates = []
        
        for img in images_tta:
            conversation = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": f"{row['question']}\nProvide the answer and a detailed explanation."}]}]
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=process_vision_info(conversation)[0], padding=True, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.01)
                
            out = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            ans = "unknown"
            exp = out
            if "Explanation:" in out:
                try:
                    parts = out.split("Explanation:")
                    ans = parts[0].replace("Answer:", "").strip().lower()
                    exp = parts[1].strip()
                except: pass
            else:
                ans = out.split("\n")[0].strip().lower()
            
            candidates.append((ans, exp))
            
        # === 投票 ===
        answers = [c[0] for c in candidates]
        most_common_ans, count = Counter(answers).most_common(1)[0]
        
        # 选解释：如果原图答案就是多数派，用原图解释；否则找第一个匹配的
        if candidates[0][0] == most_common_ans:
            final_exp = candidates[0][1]
        else:
            final_exp = next(c[1] for c in candidates if c[0] == most_common_ans)
            
        results.append({"id": row['id'], "answer": most_common_ans, "explanation": final_exp})
        
    pd.DataFrame(results)[['id', 'answer', 'explanation']].to_csv(OUTPUT_FILE, index=False)
    print(f"🏆 TTA Submission Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()