import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KwikKubeNN(nn.Module):
    def __init__(self, input_size=54, hidden_size=128, output_size=19, num_layers=3, seq_len=200):
        super(KwikKubeNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_size, seq_len * output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        x = self.output_layer(x)
        x = x.view(x.size(0), self.seq_len, -1)
        x = F.softmax(x, dim=-1)
        return x