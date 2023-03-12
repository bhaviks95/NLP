import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from network import network
from Preprocessing import PreprocessedData
import time
import os
path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    dataset = PreprocessedData()
    trainset, testset = random_split(dataset, [0.8,0.2])

    batch_size = 4
    epochs = 1
    input_size = 768
    hidden_size = 768
    L = 3
    device = torch.device("mps")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)

    model = network(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adadelta(model.parameters())
    
    t0 = time.time()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            # print statistics
            running_loss += float(loss.item())
            print('Epoch ' + str(epoch+1) + ', Batch ' + str((i+1)) + '/' + str(int(len(trainset)/batch_size)) + ', loss: ' + str(running_loss /(i+1)) + ', Time from start: ' +str(time.time()-t0),end = '\r')

        correct = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                outputs = model(test_inputs)
                correct += (torch.argmax(outputs,dim=1) == test_labels).float().sum()

                print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(trainset)/batch_size)) + ', Running test: Batch ' + str(i+1) + '/' + str(250), end='\r')

            classification_error = correct/len(testset)
            
            print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(trainset/batch_size))) + ', Test set accuracy: ' + str(classification_error.item()))
        
        print('\n')
    
        print('Training done.')

        model_name =str(path) + '/model1.pt'

        # save trained model
        torch.save(model.state_dict(), model_name)
        print('Model saved.')