import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision import transforms  # torch transforms v1
from transformers import ViTForImageClassification, ViTImageProcessor, get_cosine_schedule_with_warmup
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
import random
import json
import os
import datetime

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

NUM_EPOCHS = 40

CONFIG = {
    "model_name": "google/vit-base-patch16-224",
    "num_labels": 3,
    "batch_size": 16,
    "num_epochs": NUM_EPOCHS,
    "lr_head": 1e-3,          # learning rate for classifier head
    "lr_backbone": 1e-5,      # learning rate for unfrozen backbone layers
    "weight_decay": 0.01,
    "grad_accumulation_steps": 2, # effectively simulate 32 batch size with 16 physical
    "warmup_epochs": 2,           # warmup epochs for cosine schedule
    "unfreeze_schedule": {     # dynamically scale based on total epochs
        0: 0,                               # phase 1: only classifier
        max(1, int(NUM_EPOCHS * 0.15)): 2,  # phase 2: last 2 encoder layers (unfreeze at 15%)
        max(2, int(NUM_EPOCHS * 0.30)): 4,  # phase 3: last 4 encoder layers (unfreeze at 30%)
        max(3, int(NUM_EPOCHS * 0.45)): 12, # phase 4: all 12 encoder layers (unfreeze at 45%)
    },
    "random_state": 11,        # seed for reproducibility
    "samples_per_class": 500,  # max images per class (set None to use all available)
    "val_split": 0.2,          # 20% of selected data for validation
    # Choose which metric to use for saving the best model:
    # Options: "val_acc", "val_f1", "val_loss", "val_precision", "val_recall"
    "best_model_metric": "val_f1",
    "early_stopping_patience": max(2, int(NUM_EPOCHS * 0.10)),      # Tight 10% patience (active only after final unfreeze phase)
    "data_dir": "/opt/project/dataset/waste_classification_data/TRAIN",
    "save_path": "/opt/project/tmp/waste_model.pth",
}

# ══════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════════
# TRANSFORMS (torchvision v2)
# ══════════════════════════════════════════════════════════

train_transforms = transforms.Compose([
    # ── Spatial augmentations ──
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),    # shift up to 10%
        shear=(-10, 10),         # shear distortion
    ),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),

    # ── Color / photometric augmentations ──
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.15,
    ),
    transforms.RandomGrayscale(p=0.08),
    transforms.RandomPosterize(bits=4, p=0.1),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomEqualize(p=0.1),

    # ── Noise / blur ──
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),

    # ── Tensor conversion + normalize ──
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # ── Cutout / erasing (applied on tensor) ──
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.0)),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════
# DATASET & DATALOADERS (balanced: R=1, O=0)
# Train / Val 
# ══════════════════════════════════════════════════════════

# Load full dataset to inspect class distribution
full_dataset = datasets.ImageFolder(CONFIG["data_dir"])
class_to_idx = full_dataset.class_to_idx  # e.g. {'O': 0, 'R': 1}
print(f"Original class mapping: {class_to_idx}")
print(f"Total images: {len(full_dataset)}")

# Separate indices by class
class_indices = {}
for idx, (_, label) in enumerate(full_dataset.samples):
    class_indices.setdefault(label, []).append(idx)

for cls_name, cls_idx in class_to_idx.items():
    print(f"  {cls_name} (label={cls_idx}): {len(class_indices[cls_idx])} images")

# Balance: cap each class at samples_per_class (or smallest class if None)
max_per_class = CONFIG["samples_per_class"]
seed = CONFIG["random_state"]

if max_per_class is None:
    max_per_class = min(len(indices) for indices in class_indices.values())
else:
    max_per_class = min(max_per_class, min(len(indices) for indices in class_indices.values()))

balanced_indices = []
for cls_idx, indices in class_indices.items():
    random.seed(seed)
    sampled = random.sample(indices, max_per_class)
    balanced_indices.extend(sampled)

random.seed(seed)
random.shuffle(balanced_indices)
print(f"Balanced dataset: {len(balanced_indices)} images ({max_per_class} per class, seed={seed})")

# 2-way split: train / val
val_size = int(len(balanced_indices) * CONFIG["val_split"])
train_indices = balanced_indices[val_size:]
val_indices = balanced_indices[:val_size]

print(f"Split: {len(train_indices)} train, {len(val_indices)} val")

# Create datasets with proper transforms
train_dataset_raw = datasets.ImageFolder(CONFIG["data_dir"], transform=train_transforms)
val_dataset_raw = datasets.ImageFolder(CONFIG["data_dir"], transform=val_transforms)

train_ds = Subset(train_dataset_raw, train_indices)
val_ds = Subset(val_dataset_raw, val_indices)

dataloaders = {
    'train': DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                        shuffle=True, num_workers=0, pin_memory=True),
    'val': DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                      shuffle=False, num_workers=0, pin_memory=True),
}

# Verify: E=0, O=1, R=2 (ImageFolder assigns alphabetically)
assert class_to_idx['E'] == 0, f"Expected E=0, got {class_to_idx['E']}"
assert class_to_idx['O'] == 1, f"Expected O=1, got {class_to_idx['O']}"
assert class_to_idx['R'] == 2, f"Expected R=2, got {class_to_idx['R']}"
print(f"✓ Class mapping confirmed: E=0, O=1, R=2")
print(f"✓ Train: {len(train_ds)}, Val: {len(val_ds)}")

# ══════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════

model = ViTForImageClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=CONFIG["num_labels"],
    ignore_mismatched_sizes=True,
).to(device)

# Freeze everything initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

# ══════════════════════════════════════════════════════════
# GRADUAL UNFREEZING
# ══════════════════════════════════════════════════════════

def get_encoder_layers(model):
    """Get ViT encoder layers (ordered first → last)."""
    return model.vit.encoder.layer


def unfreeze_last_n_layers(model, n):
    """Unfreeze the last N encoder layers + classifier. Freeze everything else."""
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze last N encoder layers
    encoder_layers = get_encoder_layers(model)
    total_layers = len(encoder_layers)  # 12 for vit-base

    if n > 0:
        for layer in encoder_layers[max(0, total_layers - n):]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze layernorm after encoder
        for param in model.vit.layernorm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfroze last {n}/{total_layers} encoder layers → "
          f"{trainable:,} / {total:,} params trainable ({trainable/total*100:.1f}%)")
    return trainable


def build_optimizer(model, lr_head, lr_backbone):
    """Build optimizer with differential learning rates."""
    encoder_layers = get_encoder_layers(model)
    head_params = list(model.classifier.parameters())

    backbone_params = []
    for layer in encoder_layers:
        for param in layer.parameters():
            if param.requires_grad:
                backbone_params.append(param)

    # Include layernorm params in backbone
    for param in model.vit.layernorm.parameters():
        if param.requires_grad:
            backbone_params.append(param)

    param_groups = [
        {"params": head_params, "lr": lr_head, "name": "head"},
    ]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"}
        )

    return optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

# Label Smoothing 0.1 to penalize overconfidence
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
num_epochs = CONFIG["num_epochs"]
best_metric_name = CONFIG["best_model_metric"]

# For metrics where higher is better
metric_higher_is_better = best_metric_name in ("val_acc", "val_f1", "val_precision", "val_recall")
best_metric_val = -float('inf') if metric_higher_is_better else float('inf')
epochs_without_improvement = 0

# AMP Scaler
scaler = GradScaler()

# Initialize Optimizer (Scheduler will be set up in the main loop per-stage)
optimizer = build_optimizer(model, CONFIG["lr_head"], CONFIG["lr_backbone"])
scheduler = None # dynamically initialized when unfreezing occurs

current_unfreeze = -1  # track current unfreeze level

print(f"\nℹ Best model will be saved based on: {best_metric_name}")
print(f"  ({'higher' if metric_higher_is_better else 'lower'} is better)")

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{num_epochs - 1}")
    print(f"{'='*60}")

    # ── Gradual unfreezing check ──────────────────────────
    target_unfreeze = 0
    for trigger_epoch, n_layers in sorted(CONFIG["unfreeze_schedule"].items()):
        if epoch >= trigger_epoch:
            target_unfreeze = n_layers

    if target_unfreeze != current_unfreeze:
        print(f"→ Unfreezing schedule triggered at epoch {epoch}")
        unfreeze_last_n_layers(model, target_unfreeze)
        
        # Reset optimizer for new parameters, and start a new Cosine wave
        optimizer = build_optimizer(model, CONFIG["lr_head"], CONFIG["lr_backbone"])
        
        remaining_epochs = num_epochs - epoch
        grad_acc_steps = CONFIG.get("grad_accumulation_steps", 1)
        total_steps = (len(dataloaders['train']) // grad_acc_steps) * remaining_epochs
        warmup_steps = (len(dataloaders['train']) // grad_acc_steps) * (CONFIG.get("warmup_epochs", 2) if epoch == 0 else 1)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        current_unfreeze = target_unfreeze
        epochs_without_improvement = 0  # Reset patience for the new architecture phase

    # ── Training phase ────────────────────────────────────
    model.train()
    running_loss = 0.0
    num_train = 0
    all_train_preds = []
    all_train_labels = []

    grad_acc_steps = CONFIG.get("grad_accumulation_steps", 1)
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with Automatic Mixed Precision
        with autocast('cuda'):
            outputs = model(inputs)['logits']
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss = loss / grad_acc_steps  # Normalize loss for accumulation

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Step optimizer and scheduler every grad_acc_steps
        if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(dataloaders['train']):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step() # Cosine decays per micro-batch

        running_loss += loss.item() * inputs.size(0)
        num_train += inputs.size(0)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / num_train
    all_train_preds = np.array(all_train_preds)
    all_train_labels = np.array(all_train_labels)
    train_acc = (all_train_preds == all_train_labels).mean()
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
    train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
    train_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)

    # ── Validation phase ──────────────────────────────────
    model.eval()
    val_loss = 0.0
    num_val = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(inputs)['logits']
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            num_val += inputs.size(0)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / num_val
    all_val_preds = np.array(all_val_preds)
    all_val_labels = np.array(all_val_labels)
    val_acc = (all_val_preds == all_val_labels).mean()
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

    # Cosine scheduler steps per micro-batch in train loop, no val_loss step needed!

    # ── Logging ──────────────────────────────────────────
    current_lr_head = optimizer.param_groups[0]['lr']
    current_lr_backbone = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0

    print(f"  Train │ Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}  Prec: {train_precision:.4f}  Rec: {train_recall:.4f}")
    print(f"  Val   │ Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}  Prec: {val_precision:.4f}  Rec: {val_recall:.4f}")
    print(f"  LR    │ head: {current_lr_head:.6f}  backbone: {current_lr_backbone:.6f}")


    # ── Save best model (based on chosen metric) ────────
    epoch_metrics = {
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_loss": val_loss,
        "val_precision": val_precision,
        "val_recall": val_recall,
    }
    current_metric = epoch_metrics[best_metric_name]

    is_best = (
        current_metric > best_metric_val if metric_higher_is_better
        else current_metric < best_metric_val
    )

    if is_best:
        best_metric_val = current_metric
        torch.save(model.state_dict(), CONFIG["save_path"])
        print(f"  ✓ Best model saved ({best_metric_name}={current_metric:.4f})")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"  ! No improvement for {epochs_without_improvement} epoch(s)")

    # Early stopping check
    if CONFIG.get("early_stopping_patience") and epochs_without_improvement >= CONFIG["early_stopping_patience"]:
        if epoch >= int(NUM_EPOCHS * 0.45):
            print(f"\n✋ Early stopping triggered! Validation metric did not improve for {CONFIG['early_stopping_patience']} consecutive epochs.")
            print(f"\n{'='*60}")
            print("Final Validation Classification Report (Early Stopping):")
            print('='*60)
            target_names = ['E (0)', 'O (1)', 'R (2)']
            print(classification_report(all_val_labels, all_val_preds, target_names=target_names, digits=4))
            break
        else:
            print(f"  [i] Patience parameter hit, but delaying termination until the final unfreeze phase unlocks...")

    if epoch == num_epochs - 1:
        print(f"\n{'='*60}")
        print("Final Validation Classification Report:")
        print('='*60)
        target_names = ['E (0)', 'O (1)', 'R (2)']
        print(classification_report(all_val_labels, all_val_preds, target_names=target_names, digits=4))

print(f"\n{'='*60}")
print(f"Training complete. Best {best_metric_name}: {best_metric_val:.4f}")
print(f"Model saved to {CONFIG['save_path']}")
print(f"{'='*60}")