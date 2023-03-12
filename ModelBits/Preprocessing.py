
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os

path = os.path.dirname(os.path.abspath(__file__))

class PreprocessedData(Dataset):
    def __init__(self, pseudolabels = False, num_context = 1, tokenizer = "bert-base-cased"):
        super().__init__()
        self.tokenizer = tokenizer
        self.autotokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.df = self.extract()
        self.sentences, self.labels = self.generate_word_dataset(df=self.df, num_context=num_context, pseudolabels = pseudolabels)
        

    def __len__(self):
            return len(self.labels)
    
    def __getitem__(self, idx):
            label = self.labels[idx]
            sentences = self.sentences[idx]
            sample = [sentences, label]
            return sample
    
    def extract(self):
        df_groundtruth = pd.read_csv(str(path)+'/../ClaimBuster_Datasets/datasets/groundtruth.csv')
        df_all_sentences = pd.read_csv(str(path)+'/../ClaimBuster_Datasets/datasets/all_sentences.csv')
        df_crowdsourced = pd.read_csv(str(path)+'/../ClaimBuster_Datasets/datasets/crowdsourced.csv')
        df_crwd_grnd = pd.concat([df_groundtruth,df_crowdsourced],axis=0)
        sentence_id_with_v = df_crwd_grnd['Sentence_id'].values
        df_without_label = df_all_sentences[~((df_all_sentences['Speaker'] == 'Information') | (df_all_sentences['Speaker_title'].isna()) | (df_all_sentences['Sentence_id'].isin(sentence_id_with_v))) ]
        df_without_label['Verdict'] = 6 #all non labels have 7 as verdict
        df_full = pd.concat([df_crwd_grnd,df_without_label],axis=0)
        df_full = df_full.sort_values('Sentence_id')
        df_full['Verdict'] = df_full['Verdict'].apply(lambda x: x+1)
        text = df_full["Text"].values.tolist()
        tokenized_text = self.autotokenizer(text, padding="longest", truncation=True)['input_ids']

        self.longest = len(tokenized_text[0])
       

        df_full['Text'] = tokenized_text
        df_full.reset_index(drop=True,inplace=True)
        
        return df_full
    
    def generate_word_dataset(self,df,num_context,pseudolabels):
        dataset = []
        label = []
        for index,row in df.iterrows():
            if index == 0 or index == len(df)-num_context:
                continue
            past = df.iloc[index-1]['Text']
            curr = row['Text']
            future = df.iloc[index+1]['Text']    
            if row['File_id'] != df.iloc[index-1]['File_id']:
                past = self.autotokenizer("",max_length = self.longest,padding="max_length", truncation=True)['input_ids']
            elif row['File_id'] != df.iloc[index+1]['File_id']:
                future = self.autotokenizer("",max_length = self.longest,padding="max_length", truncation=True)['input_ids']
            dataset.append([past,curr,future])
            label.append(row['Verdict'])
            
        label = np.array(label)

        labelled_indices = np.where(label!=7)[0]
        
        label = label.tolist()

        if pseudolabels == False:
            dataset = [dataset[i] for i in labelled_indices]
            label = [label[i] for i in labelled_indices]

        return torch.tensor(dataset), torch.tensor(label)

a = PreprocessedData()
print(a.labels.unique())