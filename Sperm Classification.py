import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import os
import gdown
from zipfile import ZipFile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import random

# 1. دانلود فایل از گوگل درایو و استخراج

def download_and_extract_drive_file(file_id, extract_to):
    zip_file_path = "/tmp/sperm_data.zip"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_file_path, quiet=False)
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Data extracted to: {extract_to}")

data_dir = "/tmp/sperm_dataset"
if not os.path.exists(data_dir):
    drive_file_id = "1QAXEANHEbCf86dnUkuehiyHdvk59QOB1"
    download_and_extract_drive_file(drive_file_id, data_dir)

train_path = os.path.join(data_dir, 'train')
val_path = os.path.join(data_dir, 'val')

# 2. آماده‌سازی داده‌ها
if not os.path.exists(val_path):
    os.makedirs(val_path)
    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        if os.path.isdir(class_dir):
            val_class_dir = os.path.join(val_path, class_name)
            os.makedirs(val_class_dir, exist_ok=True)
            files = os.listdir(class_dir)
            train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
            for val_file in val_files:
                shutil.move(os.path.join(class_dir, val_file), os.path.join(val_class_dir, val_file))

# 3. تعریف دیتاست
class SpermDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, img_path

# 4. تعریف مدل
class SpermClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpermClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 5. تنظیمات مدل و داده‌ها
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SpermDataset(train_path, transform=transform)
val_dataset = SpermDataset(val_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpermClassifier(num_classes=18).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. آموزش مدل
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_accuracy = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# 7. ذخیره مدل
torch.save(model.state_dict(), "sperm_classifier.pth")

# 8. تابع پیش‌بینی
def predict(image_path, model, transform, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return pred.item()

# 9. نمایش تصاویر نمونه و پیش‌بینی
random_samples = random.sample(range(len(val_dataset)), 5)
for idx in random_samples:
    _, _, img_path = val_dataset[idx]
    predicted_class = predict(img_path, model, transform, device)
    plt.imshow(Image.open(img_path))
    plt.title(f"Predicted: {val_dataset.classes[predicted_class]}")
    plt.axis('off')
    plt.show()

# Example Usage
# prediction = predict("path_to_your_image.jpg", model, transform, device)
# print(f"Predicted class: {train_dataset.classes[prediction]}")
