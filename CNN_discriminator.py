import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._to_linear = None
        self._calculate_input_feature_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _calculate_input_feature_size(self):
        with torch.no_grad():
            self._to_linear = self.features(torch.zeros(1, 3, 128, 128)).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(self.filenames) == 0:
            raise RuntimeError("No images found in dataset directory.")
        print(f"Found {len(self.filenames)} images in dataset directory.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        label = 1 if 'real' in img_path else 0
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = CustomDataset(r'C:\Users\aayus\PycharmProjects\Discriminator_comparison\dataset\img_align_celeba\img_align_celeba', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

def train_model(num_epochs, model, dataloaders):
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:  # Adjust as necessary for different verbosity
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                print(f"Labels: {labels[:10].view(-1)}")
                print(f"Outputs: {outputs[:10].view(-1)}")

train_model(10, model, {'train': train_loader})

loss_history, accuracy_history = train_model(10, model, {'train': train_loader})

plt.plot(loss_history, label='Training Loss')
plt.plot(accuracy_history, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()


def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    # Compute the squared difference of means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # Compute the trace of the covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check for imaginary numbers and handle them
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_inception_score(preds, num_splits=10):
    scores = []
    preds = torch.nn.functional.softmax(preds, dim=1)
    for i in range(num_splits):
        part = preds[i * (preds.shape[0] // num_splits): (i + 1) * (preds.shape[0] // num_splits), :]
        kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0, keepdim=True)))
        kl = kl.sum(dim=1)
        scores.append(torch.exp(kl.mean()).item())
    return np.mean(scores), np.std(scores)

inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.eval()
if torch.cuda.is_available():
    inception_model = inception_model.cuda()

# Assuming 'images' are from your validation dataset or generated images
with torch.no_grad():
    pred = inception_model(images)[0]
    # Extract features for FID from one of the pre-auxlogits layers
    features = pred[:, :2048]

real_features = np.random.randn(100, 2048)  # Placeholder for real image features
fake_features = features.cpu().numpy()  # Your generated/fake image features

fid_score = calculate_fid(real_features, fake_features)
print('FID score:', fid_score)

is_mean, is_std = calculate_inception_score(pred)
print('Inception score:', is_mean, '±', is_std)


print("Training complete.")