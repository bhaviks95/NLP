
from transformers import AutoModel
from Preprocessing import PreprocessedData
from torch.utils.data import DataLoader

trainset = PreprocessedData()
print(trainset)
batch_size = 64
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)

sentences, labels = next(iter(trainloader))
print(sentences.shape)

model = AutoModel.from_pretrained('bert-base-cased')
print(model(sentences))