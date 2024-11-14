import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath("../models"))
from nn import *
from dataloader import *
sys.path.append(os.path.abspath("../src"))
from gen import *

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        seq = constructSeq(outputs, targets)
        seq = torch.tensor(seq).float().clone().detach().requires_grad_(True)
        targets = torch.tensor(targets).float().clone().detach().requires_grad_(True)
        loss = criterion(seq, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            seq = constructSeq(outputs, targets)
            seq = torch.tensor(seq).double().clone().detach().requires_grad_(True)
            targets = torch.tensor(targets).double().clone().detach().requires_grad_(True)
            loss = criterion(seq, targets)
            total_loss += loss.item()
            total += targets.size(0)
    return total_loss / len(dataloader)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # Pad inputs with 0
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)  # Pad targets with -1
    return inputs_padded, targets_padded

input_size = 54
hidden_size = 128
output_size = 18
num_layers = 3
batch_size = 64
learning_rate = 0.001
num_epochs = 1000

data, targets = loadData(train = True)
num_samples = len(data)

split_ratio = 0.8
split_index = int(split_ratio * num_samples)
train_data, val_data = data[:split_index], data[split_index:]
train_targets, val_targets = targets[:split_index], targets[split_index:]

train_dataset = RubiksCubeDataset(train_data, train_targets)
val_dataset = RubiksCubeDataset(val_data, val_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KwikKubeNN(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
