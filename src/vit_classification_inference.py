import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
import os
from skimage.util import view_as_windows


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=3)

# Load the saved model
model.load_state_dict(torch.load('/opt/project/tmp/vit_model1.pth'))
model = model.to(device)
model.eval()
