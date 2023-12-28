import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn as nn
from tqdm import tqdm

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 100), 
                      nn.Tanh(), 
                      nn.Linear(100, 10))

opt = torch.optim.AdamW(model.parameters(), lr=0.001)
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.85 * averaged_model_parameter + 0.15 * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model)



transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)


for i in range(1000):
    temp_loss = .0
    temp_ema_loss = .0
    temp_size = .0
    for (x,y) in training_loader:
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        loss.backward()
        # Adjust learning weights
        opt.step()
        ema_model.update_parameters(model)
        ema_loss = torch.nn.CrossEntropyLoss()(ema_model(x),y).item()
        temp_loss += loss.item()
        temp_ema_loss += ema_loss
        temp_size += x.shape[0]
    print(f"{i} epoch passed TRAIN: the loss is {temp_loss/temp_size}, ema loss is {temp_ema_loss/temp_size}")

    with torch.no_grad():
        temp_loss_v = .0
        temp_ema_loss_v = .0
        temp_size_v = .0
    
        for (x,y) in validation_loader:
            output = model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            ema_loss = torch.nn.CrossEntropyLoss()(ema_model(x),y).item()
            temp_loss_v += loss.item()
            temp_ema_loss_v += ema_loss
            temp_size_v += x.shape[0]
        print(f"{i} epoch passed VAL: the loss is {temp_loss_v/temp_size_v}, ema loss is {temp_ema_loss_v/temp_size_v}")

### Conclusion EMA is a must!!!
