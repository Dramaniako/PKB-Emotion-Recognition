import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter
from PIL import Image
import io
import copy
from tqdm import tqdm

# ----------------- CONFIGURATION -----------------
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 7

# Set to True if you want to pre-train on FER-2013 first
RUN_FER_PRETRAINING = True

# Dataset paths
FER_TRAIN_DIR = 'dataset_fer2013/train'
FER_TEST_DIR = 'dataset_fer2013/test'
RAF_TRAIN_DIR = 'dataset_rafdb/train'
RAF_TEST_DIR = 'dataset_rafdb/test'
# -------------------------------------------------

# 0. Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. Device Setup
# (Device setup and seed_everything are now handled inside main() to prevent worker process deadlocks)

# 2. Dataset Definition
class EmotionDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        # Cache bytes gambar mentah di RAM untuk hemat memori
        self.cache = [None] * len(paths)
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img_bytes = self.cache[idx]
        if img_bytes is None:
            with open(self.paths[idx], 'rb') as f:
                img_bytes = f.read()
            self.cache[idx] = img_bytes
            
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

def get_balanced_dataset(data_dir, transform, is_training=True):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {data_dir}")
        
    class_names = sorted(os.listdir(data_dir))
    paths = []
    labels = []
    
    for i, c in enumerate(class_names):
        class_dir = os.path.join(data_dir, c)
        if not os.path.isdir(class_dir):
            continue
        class_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        paths.extend(class_paths)
        labels.extend([i] * len(class_paths))
        
    print(f"\n[INFO] Dataset {data_dir} - Sampel Asli: {len(paths)}")
    
    if is_training:
        counter = Counter(labels)
        max_count = max(counter.values())
        
        balanced_paths = []
        balanced_labels = []
        
        for i in range(len(class_names)):
            class_indices = [idx for idx, label in enumerate(labels) if label == i]
            if not class_indices:
                continue
            duplicated_indices = np.random.choice(class_indices, size=max_count, replace=True)
            
            balanced_paths.extend([paths[idx] for idx in duplicated_indices])
            balanced_labels.extend([labels[idx] for idx in duplicated_indices])
            
        print(f"[INFO] Setelah Oversampling: {len(balanced_paths)} sampel (seimbang)")
        final_paths = balanced_paths
        final_labels = balanced_labels
    else:
        final_paths = paths
        final_labels = labels
        
    dataset = EmotionDataset(final_paths, final_labels, transform=transform)
    
    # num_workers=0 pada Windows untuk menghindari masalah multiprocessing
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_training,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Custom Focal Loss with Label Smoothing
class CategoricalFocalCrossentropy(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super(CategoricalFocalCrossentropy, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        ce_loss = -smooth_targets * log_probs
        probs = torch.exp(log_probs)
        focal_weight = self.alpha * ((1.0 - probs) ** self.gamma)
        
        loss = focal_weight * ce_loss
        return loss.sum(dim=-1).mean()

# 4. Model Builder
def build_model(pretrained_weights=None, trainable_backbone=False):
    # Load pretrained EfficientNet-B2
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    
    # Set backbone trainability
    for param in model.features.parameters():
        param.requires_grad = trainable_backbone
        
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )
    
    # Load custom weights if available
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"[INFO] Memuat bobot kustom dari: {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        
        model_dict = model.state_dict()
        # Filter matching weights
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        if len(pretrained_dict) < len(state_dict):
            print("[WARN] Shape mismatch pada head — memuat bobot backbone saja...")
            
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("[INFO] Bobot berhasil dimuat!")
        
    return model

# 5. Training Epoch Logic
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            
            # Gradient clipping dengan AMP
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })
        
    if scheduler:
        scheduler.step()
        
    return running_loss / total, correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / total, correct / total

def evaluate_with_tta(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Normal prediction
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=-1)
            
            # Horizontal flip prediction
            inputs_flipped = torch.flip(inputs, dims=[3]) # Horizontal flip
            outputs_flipped = model(inputs_flipped)
            probs_flipped = F.softmax(outputs_flipped, dim=-1)
            
            # Average prediction
            avg_probs = (probs + probs_flipped) / 2.0
            
            _, predicted = avg_probs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return correct / total

def fit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, checkpoint_path, device):
    # Dapatkan akurasi val awal sebelum training di fase ini
    initial_loss, initial_acc = validate_epoch(model, val_loader, criterion, device)
    best_val_acc = initial_acc
    print(f"[INFO] Akurasi val awal sebelum fase ini: {initial_acc * 100:.2f}% (loss: {initial_loss:.4f})")
    
    # Inisialisasi GradScaler untuk mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs:02d} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - lr: {lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, checkpoint_path)
            print(f"Epoch {epoch:02d}: val_accuracy meningkat menjadi {val_acc:.4f}, menyimpan model ke {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:02d}: val_accuracy tidak meningkat dari {best_val_acc:.4f}")
            
        if patience_counter >= patience:
            print(f"Early stopping dipicu setelah {epoch} epoch.")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model

# Main Pipeline
def main():
    seed_everything(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[INFO] CUDA terdeteksi. Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] GPU/CUDA tidak terdeteksi! Pelatihan akan berjalan di CPU.")
        
    fer_weights_path = 'models_archive/fer2013_efficientnetb2_pytorch.pth'
    raf_weights_path = 'samaya_rafdb_sota_pytorch_b2_adamw.pth'
    
    criterion = CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    
    # --- STEP 1: PRE-TRAINING ON FER-2013 ---
    if RUN_FER_PRETRAINING:
        print("\n=== TAHAP 1: PRE-TRAINING PADA FER-2013 ===")
        fer_train_ds = get_balanced_dataset(FER_TRAIN_DIR, train_transform, is_training=True)
        fer_val_ds = get_balanced_dataset(FER_TEST_DIR, val_transform, is_training=False)
        
        model = build_model(trainable_backbone=True)
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
        
        model = fit_model(
            model=model,
            train_loader=fer_train_ds,
            val_loader=fer_val_ds,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=15,
            patience=5,
            checkpoint_path=fer_weights_path,
            device=device
        )
        print("[INFO] Pre-training FER-2013 Selesai.")
    else:
        print("\n[INFO] Tahap 1 (Pre-training FER-2013) dilewati.")

    # --- STEP 2: FINE-TUNING ON RAF-DB ---
    print("\n=== TAHAP 2: FINE-TUNING PADA RAF-DB ===")
    raf_train_ds = get_balanced_dataset(RAF_TRAIN_DIR, train_transform, is_training=True)
    raf_val_ds = get_balanced_dataset(RAF_TEST_DIR, val_transform, is_training=False)
    
    # Load model with pre-trained weights if available
    weights_to_load = fer_weights_path if os.path.exists(fer_weights_path) else None
    model = build_model(pretrained_weights=weights_to_load, trainable_backbone=False)
    model = model.to(device)
    
    # Phase 2.1: Train Custom Head Only
    print("\n>>> Phase 2.1: Training Custom Head Only (10 Epochs)")
    optimizer_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_p1, T_0=5, T_mult=1, eta_min=1e-6)
    
    model = fit_model(
        model=model,
        train_loader=raf_train_ds,
        val_loader=raf_val_ds,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        epochs=10,
        patience=8,
        checkpoint_path=raf_weights_path,
        device=device
    )
    
    # Phase 2.2: Fine-Tuning Top 30 Layers of Backbone
    print("\n>>> Phase 2.2: Fine-Tuning Top Layers of Backbone (15 Epochs)")
    # Unfreeze blocks 7 and 8, and classifier
    for i, child in enumerate(model.features):
        if i >= 7:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
                
    optimizer_p2 = optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 5e-5, 'weight_decay': 1e-2},
        {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-2}
    ])
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=15, eta_min=1e-7)
    
    model = fit_model(
        model=model,
        train_loader=raf_train_ds,
        val_loader=raf_val_ds,
        criterion=criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        epochs=15,
        patience=8,
        checkpoint_path=raf_weights_path,
        device=device
    )
    
    # Phase 2.3: Fine-Tuning All Layers
    print("\n>>> Phase 2.3: Fine-Tuning Seluruh Model (10 Epochs)")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_p3 = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},
        {'params': model.classifier.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2}
    ])
    scheduler_p3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p3, T_max=10, eta_min=1e-8)
    
    model = fit_model(
        model=model,
        train_loader=raf_train_ds,
        val_loader=raf_val_ds,
        criterion=criterion,
        optimizer=optimizer_p3,
        scheduler=scheduler_p3,
        epochs=10,
        patience=8,
        checkpoint_path=raf_weights_path,
        device=device
    )
    
    # Evaluate with TTA
    print("\n=== EVALUASI MODEL AKHIR (Dengan Test-Time Augmentation) ===")
    tta_accuracy = evaluate_with_tta(model, raf_val_ds, device)
    print(f"[SUCCESS] Akurasi Akhir dengan TTA: {tta_accuracy * 100:.2f}%")
    print(f"[SUCCESS] Model terbaik disimpan di: '{raf_weights_path}'")

if __name__ == '__main__':
    main()
