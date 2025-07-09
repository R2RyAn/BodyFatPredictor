import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)

# Load weights
model.load_state_dict(torch.load("bodyfat_modelV2.pth", map_location=torch.device('cpu')))
model.eval()  # important!
