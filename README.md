# Explainable VQA on CLEVR-X (JAIST I491E Final Project)

This repository contains the source code for the final project of I491E. The goal is to optimize a small multimodal model (<4B) to achieve high accuracy on the CLEVR-X VQA task.

**Final Accuracy:** 0.91464

## Files Description

* `train_teacher.py`: Training the Qwen2.5-VL-3B teacher model.
* `gen_pseudo.py`: Generating pseudo-labels for unlabeled data using Self-Consistency.
* `train_student.py`: Training the final student model with 3:1 Data Balancing and Regularization.
* `inference_tta.py`: Final inference script using Multi-Scale Test-Time Augmentation (TTA).
* `run_final.sh`: The main shell script to reproduce the entire pipeline.

## How to Run

```bash
chmod +x run_final.sh
./run_final.sh
