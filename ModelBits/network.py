import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import relu
import pandas as pd
import numpy as np

from transformers import AutoModel

class network(nn.Module):
    def __init__(self, input_size, hidden_size, L = 3, bidirectional = True, num_LSTM_layers = 1, dropout = 0.2, embed = 'bert_base_cased'):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target = np.ceil(L/2)
        self.D = 2 if bidirectional == True else 1

        self.embed = AutoModel.from_pretrained(embed)
        self.embed.train()

        self.LSTM = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_LSTM_layers = num_LSTM_layers,
            bidirectional = bidirectional,
            dropout = dropout)
        self.fc1 = nn.linear(hidden_size*self.D, 192)
        self.fc2 = nn.linear(192, 3)
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.softmax(dim=1)

    def __call__(self, input):
        output = self.forward(input)
        return output
    
    def forward(self, x):
        
        x = self.LSTM(x)[self.target]
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

        
        