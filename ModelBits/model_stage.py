
from transformers import AutoModel
from Preprocessing import PreprocessedData
from torch.utils.data import DataLoader
import torch
import time

trainset = PreprocessedData()
print(trainset)
batch_size = 64
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)

sentences, labels = next(iter(trainloader))
split = torch.tensor_split(sentences,3,dim=1)
t1 = time.time()
model = AutoModel.from_pretrained('bert-base-cased')
t2 = time.time()
print(t2-t1)
print([model(s.squeeze())[0] for s in split])