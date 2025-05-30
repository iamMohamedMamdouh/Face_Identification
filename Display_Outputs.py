import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname)
                          for fname in os.listdir(image_dir)
                          if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
num_ftrs = model.fc.in_features

NUM_CLASSES = 125
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device))
model = model.to(device)
model.eval()  

class_names = [f"person_{i}" for i in range(NUM_CLASSES)]

test_dir = "D:/Coding/python/Face_Identification/face_identification/test"
dataset = TestDataset(test_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    for inputs, img_names in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for img_name, pred in zip(img_names, preds):
            print(f"{img_name}: ✅ {class_names[pred]}")
