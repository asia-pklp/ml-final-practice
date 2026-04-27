import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

CONFIG = {
    "model_name": "facebook/deit-base-distilled-patch16-224",
    "num_labels": 3,
    "batch_size": 16,
    "test_dir": "/opt/project/dataset/waste_classification_data/TEST", 
    "model_path": "/opt/project/tmp/waste_model.pth",
}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════════
# TRANSFORMS & DATASET
# ══════════════════════════════════════════════════════════

# Use the exact same validation transforms as the training script
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if not os.path.exists(CONFIG["test_dir"]):
    print(f"Test directory not found at {CONFIG['test_dir']}")
    # Fallback to lowercase 'test' just in case
    CONFIG["test_dir"] = "/opt/project/dataset/waste_classification_data/test"

print(f"Loading test dataset from: {CONFIG['test_dir']}")
test_dataset = datasets.ImageFolder(CONFIG["test_dir"], transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

class_names = test_dataset.classes
print(f"Classes found: {class_names}")

# ══════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════

model = ViTForImageClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=CONFIG["num_labels"],
    ignore_mismatched_sizes=True,
)

print(f"Loading weights from {CONFIG['model_path']}")
model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
model = model.to(device)
model.eval()

# ══════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════

all_preds = []
all_labels = []

print("Running inference...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)['logits']
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ══════════════════════════════════════════════════════════
# METRICS & PLOTTING
# ══════════════════════════════════════════════════════════

acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

cm_path = "/opt/project/tmp/waste_confusion_matrix.png"
plt.savefig(cm_path)
print(f"Confusion matrix plot saved to {cm_path}")