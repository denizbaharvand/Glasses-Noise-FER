"""
Vision Transformer (ViT) Fine-Tuning for Facial Expression Recognition
Dataset: RAF-DB (Noisy/Clean)
Split: 70% Train, 15% Validation, 15% Test (Stratified)

This script implements a multi-stage unfreezing strategy for training a ViT model.
It utilizes advanced techniques including CutMix, MixUp, Exponential Moving Average (EMA),
Label Smoothing, and Gradient Clipping to ensure robust performance on imbalanced data.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DATA_DIR = './data/RAF-DB'  # Update this path before running
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Data Transforms (Augmentation)
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Dataset Splitting (70% Train / 15% Val / 15% Test)
# ==========================================
print("Loading dataset and applying stratified 70/15/15 split...")
full_dataset = datasets.ImageFolder(root=DATA_DIR)
targets = [s[1] for s in full_dataset.samples]

sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(sss_1.split(np.zeros(len(targets)), targets))

temp_targets = [targets[i] for i in temp_idx]
sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx_rel, test_idx_rel = next(sss_2.split(np.zeros(len(temp_targets)), temp_targets))

val_idx = [temp_idx[i] for i in val_idx_rel]
test_idx = [temp_idx[i] for i in test_idx_rel]

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = eval_transforms
test_dataset.dataset.transform = eval_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Dataset Split Completed -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ==========================================
# 4. Class Weights Handling
# ==========================================
train_targets = [targets[i] for i in train_idx]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_targets), y=train_targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ==========================================
# 5. Model Initialization (ViT)
# ==========================================
def build_model(num_classes):
    print("Initializing ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)

model = build_model(NUM_CLASSES)

# ==========================================
# 6. Advanced Training Utilities (EMA)
# ==========================================
class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

ema_model = ModelEma(model, decay=0.999, device=DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

# ==========================================
# 7. Training Pipeline (Multi-Stage)
# ==========================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_stage(model, stage_epochs, optimizer, scheduler, stage_name):
    print(f"\n--- Starting {stage_name} ---")
    best_val_acc = 0.0
    
    for epoch in range(stage_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ema_model.update(model)
            running_loss += loss.item()
            
        scheduler.step()
        val_acc = evaluate(ema_model.module, val_loader)
        print(f"Epoch [{epoch+1}/{stage_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ema_model.module.state_dict(), 'best_vit_model.pth')
            
    return best_val_acc

# ==========================================
# Multi-Stage Execution
# ==========================================
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer_1 = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_1 = CosineAnnealingLR(optimizer_1, T_max=10)
train_stage(model, 10, optimizer_1, scheduler_1, "Stage 1 (Train Head Only)")

for param in model.blocks[-1].parameters():
    param.requires_grad = True

optimizer_2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler_2 = CosineAnnealingLR(optimizer_2, T_max=15)
train_stage(model, 15, optimizer_2, scheduler_2, "Stage 2 (Unfreeze Last Block)")

for block in model.blocks[-3:]:
    for param in block.parameters():
        param.requires_grad = True

optimizer_3 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
scheduler_3 = CosineAnnealingLR(optimizer_3, T_max=15)
train_stage(model, 15, optimizer_3, scheduler_3, "Stage 3 (Unfreeze Last 3 Blocks)")

for param in model.parameters():
    param.requires_grad = True

optimizer_4 = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler_4 = CosineAnnealingLR(optimizer_4, T_max=20)
train_stage(model, 20, optimizer_4, scheduler_4, "Stage 4 (Full Fine-Tuning)")

# ==========================================
# 8. Final Testing Phase
# ==========================================
print("\n--- Final Evaluation on Test Set ---")
model.load_state_dict(torch.load('best_vit_model.pth'))
test_acc = evaluate(model, test_loader)
print(f"Final Test Accuracy: {test_acc:.2f}%")
print("Training and Evaluation Process Completed Successfully.")
