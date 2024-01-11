import torch
import torch.nn as nn
import torchvision
import os
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import cv2
from PIL import Image
import torch.optim as optim
import torchsummary as summary
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# def loader_(path):
#     try:
#         img = Image.open(path)
#         img = img.convert('RGB')
#         return img
#     except Exception as e:
#         print(f"Failed to load image at path: {path}. Error: {e}")
#         return None  # 返回 None 表示失敗

train_dir = "training_dataset"
test_dir = "validation_dataset"
# RESNET50的資料集 加載
trainset = datasets.ImageFolder(root=train_dir, transform=transform)
# trainset = [(img, label) for img, label in trainset if img is not None]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.ImageFolder(root=test_dir, transform=transform_test)
# testset = [(img, label) for img, label in testset if img is not None]
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print(trainset.class_to_idx)

# for img_path, _ in trainset.samples:
#     try:
#         print(img_path)
#         img = Image.open(img_path)
#         img = img.convert('RGB')
#     except Exception as e:
#         print(f"Trainset Failed to load image at path: {img_path}. Error: {e}")
# print("check1 done")
# for img_path, _ in testset.samples:
#     try:
#         print(img_path)
#         img = Image.open(img_path)
#         img = img.convert('RGB')
#     except Exception as e:
#         print(f"Testset Failed to load image at path: {img_path}. Error: {e}")
# print("check2 done")

classes = trainset.classes
print(classes)
print(len(trainset))
print(len(testset))

resnet50=torchvision.models.resnet50()
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    resnet50.to(device)
    
print('Resnet50')
print(num_ftrs)

# 損失函數、優化器
criterion = nn.BCELoss()  # 二元交叉熵損失
optimizer = optim.Adam(resnet50.parameters(), lr=0.001) 
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []

#train
num_epochs = 100
best_val_loss = float("inf")
patience = 5
cur_patience = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch +1}/{num_epochs}")
    run_loss=0.0
    correct_train=0
    total_train=0
    resnet50.train()

    for i,data in enumerate(trainloader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
            
        outputs = resnet50(inputs)
        loss = criterion(outputs.squeeze(), labels.float())  # 二元分類 標籤擴展成二維
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        run_loss += loss.item()
        
        if (i+1) %50 == 0:
            print(f"Epoch: {epoch+1}, [{(i+1)*(len(inputs))}/{(len(trainset))}]")
        
        # 準確率
        predicted = (outputs > 0.5).float()  # 根据概率值进行预测
        total_train += labels.size(0)
        correct_train += (predicted == labels.unsqueeze(1).float()).sum().item()
    
    scheduler.step()
    train_accuracy.append(100 * correct_train / total_train)
    train_loss.append(run_loss / len(trainloader))
    
    print(f"Train Loss: {train_loss[-1]:.4f} Train Accuracy: {train_accuracy[-1]:.2f}%")
    
    # Validation
    resnet50.eval()
    correct_val = 0
    total_val = 0
    run_loss_val = 0.0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            outputs = resnet50(inputs)
            
            loss = criterion(outputs.squeeze(), labels.float())
            run_loss_val += loss.item()
            
            # 準確率
            predicted = (outputs > 0.5).float()  # 根据概率值进行预测
            total_val += labels.size(0)
            correct_val += (predicted == labels.unsqueeze(1).float()).sum().item()

    
    val_accuracy.append(100 * correct_val / total_val)
    val_loss.append(run_loss_val / len(testloader))
    
    print(f"Validation Loss: {val_loss[-1]:.4f} Validation Accuracy: {val_accuracy[-1]:.2f}%")
    
    # Early Stopping
    if (val_loss[-1] < best_val_loss):
        best_val_loss = val_loss[-1]
        cur_patience = 0
    else:
        cur_patience += 1
        if cur_patience >= patience and epoch >= 39:
            print(f'Early stop after {epoch + 1} epoch')
            break

print('Done')
print('Save the files')        
np.savetxt('train_loss.txt', train_loss)
np.savetxt('val_loss.txt', val_loss)

np.savetxt('train_accuracy.txt', train_accuracy)
np.savetxt('val_accuracy.txt', val_accuracy)

torch.save(resnet50, 'Wei_ResNet50-erasing_weights.pth')