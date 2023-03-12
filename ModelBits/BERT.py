# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

cwd = "/Users/adnanoomerjee/Library/CloudStorage/OneDrive-Personal/Uni work/COMP0087 NLP/NLP/"
os.chdir(cwd)
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from sklearn.metrics import accuracy_score




# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess(text,max_seq_length):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return inputs['input_ids'], inputs['attention_mask']

# %%
train_texts = ['This is a positive example.', 'This is a negative example.']
train_labels = [1, 0]

# for text in train_texts:
#     print(preprocess(text,128))

# %%
df = pd.read_csv('ClaimBuster_Datasets/datasets/groundtruth.csv')
df.head()

# %%
training_text = df['Text'].values
labels = df['Verdict'].values

# %%
training_text

# %%
train_input_ids = []
train_attention_masks = []
for text in training_text:
    inputs_id,input_attention = preprocess(text,128)
    train_input_ids.append(inputs_id)
    train_attention_masks.append(input_attention)
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)


# %%
model = BertModel.from_pretrained('bert-base-uncased', num_labels=3)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Train the model
batch_size = 2
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    outputs = model(train_input_ids,train_attention_masks)
    outputs = torch.argmax(outputs,dim=1)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

train_loss /= len(train_labels) // batch_size

# %%


# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the BERT model architecture
class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        # self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc = nn.Linear(hidden_size, num_labels)    

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        # logits = F.relu(self.fc2(pooled_output))
        logits = self.fc(logits)
        return logits

# Define the optimizer and loss function
model = BertClassifier(hidden_size=768, num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Example training data
train_texts = ['This is a positive example.', 'This is a negative example.']
train_labels = [1, 0]

# Tokenize the texts and convert to input features
train_input_ids = []
train_attention_masks = []
for text in train_texts:
    inputs_id,input_attention = preprocess(text,128)
    train_input_ids.append(inputs_id)
    train_attention_masks.append(input_attention)
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)





# %%
outputs = model(train_input_ids,train_attention_masks)




