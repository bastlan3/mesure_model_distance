import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = f"{self.root_dir}/{self.annotations.iloc[idx, 0]}"
        image = Image.open(img_name).convert("RGB")
        label1 = self.annotations.iloc[idx, 1]
        label2 = self.annotations.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, (label1, label2)

def modify_resnet50(device):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    # Modify the fully connected layer to output two sets of classes
    class CustomResNet50(nn.Module):
        def __init__(self, original_model):
            super(CustomResNet50, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.fc1 = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 2)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            out1 = self.fc1(x)
            out2 = self.fc2(x)
            return out1, out2
        
        def load_model_from_checkpoint(self, checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            return self

    model = CustomResNet50(model).to(device)
    return model

    

# Define the necessary functions
def get_dataloader(csv_file, root_dir, batch_size, transform):
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Define the necessary variables
root_dir = './../PrimateRobustness'
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
import os
path = './'
model_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
model_names_filtered = [f for f in model_names if '0.05_epochs_2.pth']
for model_name in model_names_filtered:
    # Initialize the model, criterion, and device
    model = modify_resnet50(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the test dataset
    test_csv_file = './../PrimateRobustness/test_data.csv'
    test_dataloader = get_dataloader(test_csv_file, root_dir, batch_size, transform)
    # Load the model checkpoint
    model.load_model_from_checkpoint(model_name)
    # Evaluation function
    def evaluate_model(model, dataloader, criterion1, criterion2, device):
        model.eval()
        total_loss = 0.0
        correct1 = 0
        correct2 = 0
        total = 0
    
        with torch.no_grad():
            for images, (labels1, labels2) in dataloader:
                images, labels1, labels2 = images.to(device), labels1.to(device), labels2.to(device)
                outputs1, outputs2 = model(images)
                loss1 = criterion1(outputs1.float(), labels1.float())
                loss2 = criterion2(outputs2.float(), labels2.float())
                total_loss += (loss1.item() + loss2.item())
    
                _, predicted1 = torch.max(outputs1.data, 1)
                correct1 += (predicted1 == labels1.argmax(dim=1)).sum().item()
    
                correct2 += torch.sum(torch.abs(outputs2 - labels2)).item()
                total += labels1.size(0)
    
        accuracy1 = correct1 / total
        accuracy2 = correct2 / total
        avg_loss = total_loss / len(dataloader)
    
        return avg_loss, accuracy1, accuracy2
    
    # Evaluate the model
    test_loss, test_accuracy1, test_accuracy2 = evaluate_model(model, test_dataloader, criterion1, criterion2, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy 1: {test_accuracy1:.4f}")
    print(f"Test Accuracy 2: {test_accuracy2:.4f}")  
    print('model: ',model_name)