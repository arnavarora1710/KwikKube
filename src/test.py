import warnings
warnings.filterwarnings("ignore")
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

def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            outputs = outputs.view(-1, 19)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

input_size = 54
hidden_size = 128
output_size = 19
seq_len = 200
num_layers = 100
batch_size = 64
learning_rate = 3e-5
num_epochs = 1000

test_data, test_targets = loadData(train = False)
num_samples = len(test_data)

test_dataset = RubiksCubeDataset(test_data, test_targets)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KwikKubeNN(input_size, hidden_size, output_size, num_layers, seq_len).to(device)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    test_loss = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")