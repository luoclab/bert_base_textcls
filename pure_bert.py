import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import config
from transformers import AutoModelForSequenceClassification

num_filters = config.num_filters
bert_model = config.bert_model
hidden_size = config.hidden_size
n_class = config.n_class

encode_layer = config.encode_layer
filter_sizes = config.filter_sizes



# model
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(bert_model, output_hidden_states=True, return_dict=True,num_labels=config.n_class)

    def forward(self, X):
        # input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(**X)  # 返回一个output字典

        return outputs.logits

if __name__=="__main__":
    model_cls=Bert()
    tokenizer = AutoTokenizer.from_pretrained("bert_model/bert-base-chinese")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(inputs)
    logits=model_cls(inputs)
    print(logits.shape)
