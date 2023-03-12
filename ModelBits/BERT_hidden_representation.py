import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertHidden(nn.Module):
    def __init__(self):
        super(BertHidden, self).__init__()
        
        # initialize the BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        
        
        # add a linear layer to predict the output class
        # self.linear = nn.Linear(self.bert_model.config.hidden_size, num_classes)
        
    def forward(self, x):
        # tokenize each word in the input
        
        res= []

        tokens = self.tokenizer.encode(input, add_special_tokens=True)
        print(tokens)
        for input in inputs:
            tokenized_inputs = [self.tokenizer.encode(inp, add_special_tokens=True) for inp in input]

            # # pad the tokenized inputs to the same length
            max_len = max([len(inp) for inp in tokenized_inputs])
            padded_inputs = torch.zeros(len(input), max_len).long()
            for i, inp in enumerate(tokenized_inputs):
                padded_inputs[i, :len(inp)] = torch.tensor(inp)
            
            # pass the padded inputs through the BERT model
            outputs = self.bert_model(padded_inputs)
            
            # get the output of the [CLS] token
            pooled_output = outputs[1]
            res.append(pooled_output)
        

        return res