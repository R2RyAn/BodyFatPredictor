import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
from bodyfat_model import model  # <-- THIS FIXES IT!

# Load image with OpenCV
img_cv = cv2.imread("YOURIMAGELINK")  # BGR format

if img_cv is None:
    raise FileNotFoundError("Image not found!")

# Convert BGR → RGB
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Convert to PIL for TorchVision transforms
img_pil = Image.fromarray(img_rgb)

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# Transform the image
input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dim

# Predict
with torch.no_grad():
    output = model(input_tensor)  # <-- this calls the actual ResNet!
    predicted_bf = output.item()

print(f"Predicted Body Fat: {predicted_bf:.2f}%")
