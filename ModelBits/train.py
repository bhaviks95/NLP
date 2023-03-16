import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from network1 import network
import time
import os
from get_dataset import Get_Dataset

path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    def device():
        if torch.backends.mps.is_available():
            device = torch.device("mps") 
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    trainset = Get_Dataset(train=True)
    testset = Get_Dataset(train=False)
    
    batch_size = 4
    epochs = 10
    input_size = 768
    hidden_size = 768
    L = 3
    device = device()

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True, drop_last = True)

    model = network(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)

    #checkpoint = torch.load(str(path) + '/Model/model1_checkpoint_epoch_5.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])

    def train():
    ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch =- 1)
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
                print('Epoch ' + str(epoch+1) + ', Batch ' + str((i+1)) + '/' + str(int(len(trainset)/batch_size)) + ', loss: ' + str(running_loss /(i+1)) + ', Time from start: ' +str(time.time()-t0), end = '\r')
            scheduler.step()
            print('')
            correct = 0.0
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    test_inputs, test_labels = data
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    outputs = model(test_inputs)
                    correct += (torch.argmax(outputs,dim=1) == test_labels).float().sum()

                    print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(trainset)/batch_size)) + ', Running test: Batch ' + str(i+1) + '/' + str(int(len(testset)/batch_size)), end='\r')

                classification_error = correct/len(testset)
                
                print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(trainset)/batch_size)) + ', Test set accuracy: ' + str(classification_error.item()))
                
            print('')
        
            print('Training done.')

            model_name =str(path) + '/Model/model1_checkpoint_epoch_' +str(epoch+1) +'.pt'

            # save trained model
            #torch.save(model.state_dict(), model_name)

            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, model_name)
            
            print('Model saved.')

    train()