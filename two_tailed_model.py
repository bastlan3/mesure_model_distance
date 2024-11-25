import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import re
import wandb
import gc
import os
from torch.amp import autocast, GradScaler


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["WANDB_API_KEY"] = 'de19c5a1d964820317c7733e7aca31fda17e4acd'

gc.collect()
torch.cuda.empty_cache()
# Memory management and device settings
def setup_environment():
    # Clear any existing cached memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable memory efficient optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    # Set memory allocation settings
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
        torch.cuda.set_device(0)  # Set primary GPU if multiple available
        
    # Enable TF32 tensor cores on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

setup_environment()

# Function to move data to appropriate device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
# Custom Dataset
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


# DataLoader
def get_dataloader(csv_file, root_dir, batch_size, transform):
    class CustomDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.annotations = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            
            img_name = f"{self.root_dir}/{self.annotations.iloc[idx, 8]}"
            try:
                image = Image.open(img_name).convert("L").convert("RGB")
            except:
                print(img_name)
            label1 = torch.tensor([self.annotations.iloc[idx, 3], abs(1 - self.annotations.iloc[idx, 3])])* self.annotations.iloc[idx, 7].squeeze()
            label2 =  self.annotations.iloc[idx, 3]
            if self.transform:
                image = self.transform(image)
            
            return image, (label1, label2)

    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
    return dataloader

# Model Modification
def modify_resnet50(device):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    # Modify the fully connected layer to output two sets of classes
    class CustomResNet50(nn.Module):
        def __init__(self, original_model):
            super(CustomResNet50, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            for param in self.features.parameters():
                param.requires_grad = False

            self.fc1 = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2),
                nn.Sigmoid()
                )
            
            self.fc2 = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
                )
            
            self.linear = nn.Linear(258, 1)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            out1 = (self.fc1(x)-0.5)*8
            out2 = self.fc2(x)
            out2_expanded = torch.cat((out2, abs(1 - out2)), dim=1)
            out1 = out1 * out2_expanded
            #out2 = nn.Tanh()(self.linear(torch.cat((x, out1), dim=1)))
            return out1, out2   # allows for distance to be between -4 and 4
        
        def freeze(self, freeze = True):
            freeze = not freeze # Invert the freeze parameter to transorm it in require grad
            for param in self.features.parameters():
                param.requires_grad = freeze

                
        def freeze_fc2(self, freeze = True):
            freeze = not freeze # Invert the freeze parameter to transorm it in require grad
            for param in self.fc2.parameters():
                param.requires_grad = freeze

    model = CustomResNet50(model).to(device)
    return model
    


    # Parameters
csv_file = 'clean_images_data.csv'
root_dir = './../PrimateRobustness'
batch_size = 100

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model
model = nn.DataParallel(modify_resnet50(device))

# Training parameters
num_epochs = 30
initial_lr = 0.001
min_lr = 0.0001
decay_factor = 0.1
patience = 3 

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizer1 = optim.Adam(model.parameters(), lr=initial_lr)
optimizer2 = optim.Adam(model.parameters(), lr=initial_lr)

lr_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=decay_factor, patience=patience, min_lr=min_lr)
lr_scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=decay_factor, patience=patience, min_lr=min_lr)
learning_rate = initial_lr

wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="two_tailed",

    # track hyperparameters and run metadata
    config={
    "initial_learning_rate": learning_rate,
    "min_learning_rate": min_lr,
    "epochs": num_epochs,
    }
)

# DataLoader
dataloader = get_dataloader(csv_file, root_dir, batch_size, transform)

print('start_training loop', csv_file)
# Training loop
scaler = GradScaler(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_num = 0
    for images, (labels1, labels2) in dataloader:
        batch_num += 1
        # Forward pass
        with autocast(device):
            outputs = model(images.to(device))
            outputs1 = outputs[0]
            outputs2 = outputs[1].squeeze()
            
            loss1 = criterion1(outputs1.float(), labels1.float().to(device))
            loss2 = criterion2(outputs2.float(), labels2.float().to(device))
        #if epoch<1:
        model.module.freeze(False)
        optimizer1.zero_grad()
        scaler.scale(loss1).backward(retain_graph = True)
        scaler.step(optimizer1)
        scaler.update()
        lr_scheduler1.step(loss1.item())
        model.module.freeze(True)
        optimizer2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.step(optimizer2)
        scaler.update()
        lr_scheduler2.step(loss2.item())
        current_loss = loss1.item() + loss2.item()
        running_loss += current_loss
        
        
        gc.collect()
        torch.cuda.empty_cache()
        wandb.log({"loss1": loss1.item(), "loss2": loss2.item()})

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
batch_size = 20

csv_file = './../PrimateRobustness/train_data.csv'
dataloader = get_dataloader(csv_file, root_dir, batch_size, transform)

# Training parameters
num_epochs = 3
initial_lr = 0.001
min_lr = 0.0001
decay_factor = 0.1
patience = 3 
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
optimizer1 = optim.Adam(model.parameters(), lr=initial_lr)
optimizer2 = optim.Adam(model.parameters(), lr=initial_lr)

lr_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=decay_factor, patience=patience, min_lr=min_lr)
lr_scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=decay_factor, patience=patience, min_lr=min_lr)
learning_rate = initial_lr


gc.collect()
torch.cuda.empty_cache()
# Training loop
scaler = GradScaler(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_num = 0
    for images, (labels1, labels2) in dataloader:
        batch_num += 1
        # Forward pass
        with autocast(device):
            model.module.freeze(False)
            model.module.freeze_fc2(False)
            outputs = model(images.to(device))
            outputs1 = outputs[0]
            outputs2 = outputs[1].squeeze()
            
            loss1 = criterion1(outputs1.float(), labels1.float().to(device))
            loss2 = criterion2(outputs2.float(), labels2.float().to(device))
        if batch_num>1000:
            model.module.freeze_fc2(True)
            
            optimizer1.zero_grad()
            loss1.backward(retain_graph = True)
            optimizer1.step()
            model.module.freeze(True)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            model.module.freeze(True)
            optimizer2.zero_grad()
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            lr_scheduler2.step(loss2.item())
            gc.collect()
            torch.cuda.empty_cache()

        if batch_num%3000==0:
            initial_lr = initial_lr*5
            optimizer1 = optim.Adam(model.parameters(), lr=initial_lr)


        current_loss = loss1.item() + loss2.item()
        running_loss += current_loss
        
        
        gc.collect()
        torch.cuda.empty_cache()
        wandb.log({"loss1": loss1.item(), "loss2": loss2.item()})

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

print("Finished Training")

# Save the trained model
model_path = "./trained_two_tailed_resnet50_lr_{}_epochs_{}.pth".format(learning_rate, num_epochs)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
# Load the saved model
saved_model = modify_resnet50(device)
saved_model = nn.DataParallel(saved_model)
saved_model.load_state_dict(torch.load(model_path))
saved_model.eval()

test_csv_file = './../PrimateRobustness/test_data.csv'
test_dataloader = get_dataloader(test_csv_file, root_dir, batch_size, transform)

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
            print(outputs2)
            loss1 = criterion1(outputs1.float(), labels1.float())
            loss2 = criterion2(outputs2.squeeze().float(), labels2.float())
            total_loss += (loss1.item() + loss2.item())

            _, predicted1 = torch.max(outputs1.data, 1)
            correct1 += (predicted1 == labels1.argmax(dim=1)).sum().item()

            correct2 += torch.sum(torch.abs(outputs2 - labels2)).item()
            total += labels1.size(0)
            print(correct1, correct2, total)
    accuracy1 = correct1 / total
    accuracy2 = correct2 / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy1, accuracy2

# Evaluate the model
test_loss, test_accuracy1, test_accuracy2 = evaluate_model(saved_model, test_dataloader, criterion1, criterion2, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy 1: {test_accuracy1:.4f}")
print(f"Test Accuracy 2: {test_accuracy2:.4f}")  

wandb.log({"testloss": test_loss, "accuracy1": test_accuracy1, "accuracy2": test_accuracy2, "lr": learning_rate})
wandb.finish()