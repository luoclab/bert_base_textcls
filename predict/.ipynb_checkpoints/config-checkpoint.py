# # 定义一些参数，模型选择了最基础的bert中文模型
batch_size = 32
epochs = 60
bert_model = '/home/kas/kas_workspace/gongyutian/classification/bert_model/bert-base-chinese'
textcls_model = '/home/kas/kas_workspace/gongyutian/classification/model/model_v1.0_231208_3_64_80_le6_345_3.pth'
hidden_size = 768
num_filters = 3
maxlen = 512
encode_layer = 12
# filter_sizes = [2, 3, 4]
filter_sizes = [3, 4 ,5]
learning_rate = 1e-3
weight_decay = 1e-2


n_class = 7
label2idx = {'党政公文': 0, '合同文档': 1, '简历': 2, '教程指南': 3, '论文文献': 4, '总结汇报': 5, '其他类别': 6}
idx2label = {0: '党政公文', 1: '合同文档', 2: '简历', 3: '教程指南', 4: '论文文献', 5: '总结汇报', 6: '其他类别'}

# n_class = 8
# label2idx = {'党政公文': 0, '合同文档': 1, '简历': 2, '教程指南': 3, '论文文献': 4, '总结汇报': 5, '新闻资讯': 6, '其他类别': 7}
# idx2label = {0: '党政公文', 1: '合同文档', 2: '简历', 3: '教程指南', 4: '论文文献', 5: '总结汇报', 6: '新闻资讯', 7: '其他类别'}