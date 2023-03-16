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
    def __init__(self, input_size, hidden_size, L = 3, bidirectional = True, num_LSTM_layers = 1, dropout = 0.2, embed = 'bert-base-cased', batch_size = 64):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = L
        self.target = int(np.ceil(L/2)-1)
        self.D = 2 if bidirectional == True else 1
        self.batch_size = batch_size

        print("Loading BERT Model")
        self.embed = AutoModel.from_pretrained(embed)
        self.embed.train()
        print("Model Loaded")

        self.LSTM = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers = num_LSTM_layers,
            bidirectional = bidirectional,
            batch_first=True)
        self.fc1 = nn.Linear(hidden_size*self.D, 192)
        self.fc2 = nn.Linear(192, 3)
        self.dropout = torch.nn.Dropout(dropout)
        

    def __call__(self, input):
        output = self.forward(input)
        return output
    
    def forward(self, x):
        x = torch.tensor_split(x, self.L, dim = 1)
        x = torch.cat([self.embed(xi.squeeze())[1].reshape([self.batch_size,1,self.input_size]) for xi in x],dim=1)
        #x = x.reshape(self.batch_size*self.L,x.shape[-1])
        #x = self.embed(x.squeeze())[1].reshape(self.batch_size, self.L, 768)
        x = self.LSTM(x)[0][:,self.target,:]
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

        
        