import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from network import network

trainset = PreprocessedDataSet('dataset')

batch_size = 64
epochs = 10
input_size = 768
hidden_size = 768,
L = 3
device = torch.device("mps")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)

model = network(input_size=input_size, hidden_size=hidden_size, L = L).to(device)

## loss and optimiser
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adadelta(model.parameters())

## train
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        print('Epoch ' + str(epoch+1) + ', Batch ' + str(i+1) + ', loss: ' + str(running_loss /(i+1)),end = '\r')

    print('\n')
    
