import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class KwikKubeNN(nn.Module):
    def __init__(self, input_size=54, hidden_size=128, output_size=18, num_layers=3):
        super(KwikKubeNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x