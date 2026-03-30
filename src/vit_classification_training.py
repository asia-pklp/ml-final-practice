import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
import os
import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/opt/project/dataset/XXX'  # Update this path
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']} #, 'val'
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
                             shuffle=True, num_workers=4)
               for x in ['train']} #, 'val'

# Initialize the feature extractor and the model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=3).to(device)

# Freeze all the layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer: classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training function
def train_model(model, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)['logits']
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])

        print(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {datetime.datetime.now()}')

    return model

# Train the model
model = train_model(model, criterion, optimizer, num_epochs=100)

# Save the model
torch.save(model.state_dict(), '/opt/project/tmp/vit_model1.pth')
print("Model saved to vit_model1.pth")
