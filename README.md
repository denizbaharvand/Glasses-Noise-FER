# Glasses-Noise-FER
# Impact of Simulated Glasses Noise on Facial Emotion Recognition with Deep Learning Models

This repository contains the official PyTorch implementations for the research paper:  
**"Impact of Simulated Glasses Noise on Facial Emotion Recognition with Deep Learning Models".**

---

## Overview
Facial Emotion Recognition (FER) is essential in modern human–computer interaction systems. However, real-world occlusions such as eyeglasses introduce significant noise that degrades accuracy.  
This repository provides:

- A pipeline for simulating *glasses noise* on facial datasets  
- Training and evaluation of:
  - Vision Transformer (ViT-b/16)
  - Swin Transformer (Swin-T)
  - ResNet-50 with CBAM  
- Experiments conducted on the RAF-DB dataset augmented with synthetic glasses noise

---

## Repository Structure
The project is fully implemented using modular Python scripts:

- `train-vit.py` — Vision Transformer training & evaluation  
- `train_Swin-t.py` — Swin Transformer (Tiny)  
- `train_resnet50_cbam.py` — ResNet-50 + CBAM  
- `generate_dataset.py` — Simulates glasses noise and prepares noisy RAF-DB

All scripts can be executed independently.

---

## Dataset Access

The modified **RAF-DB (Noisy)** dataset is stored externally on Google Drive due to GitHub file size limits.

### Download:
Google Drive folder (public):  
https://drive.google.com/drive/folders/1q5MoDEVzyRNTZ1OMCJ_oVAvPpiVUKCzE

### Setup Instructions:
1. Download the ZIP file from the link above.  
2. Place it in the **root directory** of this repository.  
3. The training scripts will automatically:
   - Detect the ZIP  
   - Extract the dataset  
   - Prepare train/test sets

---



## Usage

To train the models, use:
```bash
python train-vit.py
python train_Swin-t.py
python train_resnet50_cbam.py

---

## Requirements

Install the required packages using:
```bash
pip install torch torchvision timm scikit-learn numpy pandas matplotlib seaborn


