# -*- coding: utf-8 -*-            
# @Author : gongyutian@wps.cn
# @Time : 2023/10/31 上午8:52
import os
from random import shuffle
import torch.utils.data as Data
from transformers import AutoModel, AutoTokenizer
from . import config

bert_model = config.bert_model
label2idx = config.label2idx
# laber2idx = {'简历': 0, '试卷': 1}
maxlen = config.maxlen

class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True, ):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.sentences[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids