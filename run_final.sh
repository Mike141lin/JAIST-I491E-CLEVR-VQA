#!/bin/bash
set -e

# 指定 Python 解释器路径
PYTHON="/home/ubuntu/miniconda3/envs/clevr_env/bin/python"
LOG_FILE="final_run_3b.log"

echo "=== 🚀 启动 Qwen2.5-3B 终极方案 ===" | tee -a $LOG_FILE

# 1. 训练 Teacher (3B)
echo "[1/4] Training Teacher..." | tee -a $LOG_FILE
$PYTHON train_teacher.py

# 2. 生成伪标签 (3B)
echo "[2/4] Generating Pseudo Labels..." | tee -a $LOG_FILE
$PYTHON gen_pseudo.py

# 3. 训练 Student (这里调用的是防过拟合版)
# 注意：如果你想复现 0.91464，这里应该改为 train_student.py
echo "[3/4] Training Final Student (Weighted Data)..." | tee -a $LOG_FILE
$PYTHON train_student_final.py

# 4. TTA 推理 (3 Scales)
echo "[4/4] Running Multi-Scale TTA Inference..." | tee -a $LOG_FILE
$PYTHON inference_tta.py

echo "🎉 终极任务完成! 请下载 submission_tta.csv" | tee -a $LOG_FILE