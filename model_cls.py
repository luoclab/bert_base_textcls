import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import config

num_filters = config.num_filters
bert_model = config.bert_model
hidden_size = config.hidden_size
n_class = config.n_class

encode_layer = config.encode_layer
filter_sizes = config.filter_sizes

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, n_class, bias=True)
        self.bias = nn.Parameter(torch.ones([n_class]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])
        # print(self.filter_list)

    def forward(self, x):
        # x: [bs, seq, hidden]
        x = x.unsqueeze(1)  # [bs, channel=1, seq, hidden]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size=(encode_layer - filter_sizes[i] + 1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes))  # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])

        output = self.Weight(h_pool_flat)
        # + self.bias  # [bs, n_class]

        return output

from transformers import AutoModelForSequenceClassification
# model
class Bert_Blend_CNN(nn.Module):
    def __init__(self):
        super(Bert_Blend_CNN, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model, output_hidden_states=True, return_dict=True)
        # self.bert = AutoModelForSequenceClassification.from_pretrained(bert_model, output_hidden_states=True, return_dict=True)

        self.linear = nn.Linear(hidden_size, n_class)
        self.textcnn = TextCNN()

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        # print("cls_embeddings.shape",cls_embeddings.shape)
        logits = self.textcnn(cls_embeddings)
        # print(" logits", logits.shape)
        return logits
