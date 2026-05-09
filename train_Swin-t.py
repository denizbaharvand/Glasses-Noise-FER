import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np

# ==========================================
# 1. Setup and Data Extraction
# ==========================================
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths (Relative paths for GitHub usability)
zip_file = './RAF-DB-noisy.zip'
extract_folder = './RAF-DB-noisy/'
data_dir = './RAF-DB-noisy/train'

# Extract dataset if not already extracted
if not os.path.exists(extract_folder):
    if os.path.exists(zip_file):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print("Extraction completed.")
    else:
        print(f"Warning: {zip_file} not found. Please ensure the dataset is in the correct directory.")
else:
    print("Dataset folder already exists.")

# ==========================================
# 2. Data Augmentation Methods (CutMix & MixUp)
# ==========================================
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data, shuffled_targets = data[indices], targets[indices]
    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = max(0.4, min(0.6, lambda_)) # Constrain lambda
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lambda_)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lambda_ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, targets.long(), shuffled_targets.long(), lambda_

def rand_bbox(size, lambda_):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lambda_)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data, shuffled_targets = data[indices], targets[indices]
    lambda_ = np.random.beta(alpha, alpha)
    lambda_ = max(lambda_, 1 - lambda_)
    data = lambda_ * data + (1 - lambda_) * shuffled_data
    return data, targets.long(), shuffled_targets.long(), lambda_

# ==========================================
# 3. Data Transformations and Loaders
# ==========================================
# Define transformations for training, validation, and testing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(15),
        transforms.RandomErasing(p=0.6, scale=(0.05, 0.25)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 0.5)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load dataset and perform 70-15-15 split
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Apply respective transforms to validation and test sets
val_dataset.dataset.transform = data_transforms['val_test']
test_dataset.dataset.transform = data_transforms['val_test']

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

print(f"Dataset Loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ==========================================
# 4. Model Architecture (Swin Transformer)
# ==========================================
# Load pre-trained Swin-T model
model = models.swin_t(weights='IMAGENET1K_V1')

# Freeze early layers, fine-tune later blocks and head
for param in model.parameters():
    param.requires_grad = False
for param in model.features[2:].parameters():  
    param.requires_grad = True
for param in model.head.parameters():
    param.requires_grad = True

# Modify final classifier for 7 emotion classes
model.head = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.head.in_features, 7)
)
model = model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003, weight_decay=0.02)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ==========================================
# 5. Training and Evaluation Functions
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=7, use_cutmix=True, use_mixup=True):
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = './best_model_swin.pth'

    for epoch in range(num_epochs):
        model.train()
        train_correct, train_total = 0, 0
        train_preds, train_labels = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply CutMix or MixUp randomly
            if use_cutmix and np.random.rand() < 0.7:
                inputs, targets_a, targets_b, lambda_ = cutmix(inputs, labels)
                outputs = model(inputs)
                loss = lambda_ * criterion(outputs, targets_a) + (1 - lambda_) * criterion(outputs, targets_b)
            elif use_mixup and np.random.rand() < 0.7:
                inputs, targets_a, targets_b, lambda_ = mixup(inputs, labels)
                outputs = model(inputs)
                loss = lambda_ * criterion(outputs, targets_a) + (1 - lambda_) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Calculate metrics
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}")

        # Update learning rate scheduler
        scheduler.step(val_acc)

        # Save best model and Early Stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

def evaluate_test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    preds, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    test_f1 = f1_score(labels_list, preds, average='weighted')
    print(f"\nFinal Test Evaluation | Accuracy: {test_acc:.2f}% | F1-Score: {test_f1:.2f}")
    return test_acc

# ==========================================
# 6. Execution
# ==========================================
if __name__ == '__main__':
    # Start training
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=7)

    # Load the best weights and evaluate on unseen test set
    model.load_state_dict(torch.load('./best_model_swin.pth'))
    test_acc = evaluate_test(model, test_loader)
