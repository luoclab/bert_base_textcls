import torch
import torch.nn as nn
import torch.utils.data as Data
import config
from dataloader import get_data_v1_0
from model_cls import Bert_Blend_CNN, TextCNN
import torch.optim as optim
from dataloader import MyDataset
import datetime
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import model_cls
from tqdm import tqdm

print(torch.cuda.is_available())
batch_size = config.batch_size#64
epochs = config.epochs
idx2label = config.idx2label
learning_rate = config.learning_rate
weight_decay = config.weight_decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_curve = []


def train(sentences_t, labels_t, model_path):
    min_sum = 1000000
    bert_blend_cnn = Bert_Blend_CNN()
    # bert_blend_cnn = nn.DataParallel(bert_blend_cnn) #最基础的bert
    bert_blend_cnn.to(device)
    optimizer = optim.Adam(bert_blend_cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_ = Data.DataLoader(dataset=MyDataset(sentences_t, labels_t), batch_size=batch_size, shuffle=True,
                             num_workers=1)
    # train
    sum_loss = 0
    total_step = len(train_)
    print(epochs, batch_size)
    # pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        for i, batch in enumerate(train_):
            optimizer.zero_grad()
            batch = tuple(p.to(device) for p in batch)
            pred = bert_blend_cnn([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
            # if epoch % 10 == 0:
            timenow = datetime.datetime.now()
            print(
                '{}:[{}|{}] step:{}/{} loss:{:.4f}'.format(timenow, epoch + 1, epochs, i + 1, total_step, loss.item()))
        print('epoch:{},loss:{}'.format(epoch + 1, sum_loss))
        train_curve.append(sum_loss)
        if epoch > 15:
            if sum_loss < min_sum:
                min_sum = sum_loss
                torch.save(bert_blend_cnn.state_dict(), model_path)
                print('save...', epoch + 1)

        train_curve.append(sum_loss)
        sum_loss = 0



if __name__ == '__main__':
    model_path = 'checkpoint/model_v2.0_np_250214_128_40_le5_10_1.pth'#np==not parallel
    train_flag = True
    if train_flag:
        trainpath = 'data/train/devide_V2.0_0.1_1/train'
        print(trainpath, model_path)
        sentences_train, labels_train, _ = get_data_v1_0(trainpath)
        ## split_idx = int(len(sentences)/10)
        # sentences_test, labels_test,sentences_train, labels_train = sentences[0:split_idx], labels[0:split_idx],sentences[split_idx:], labels[split_idx:]
        train(sentences_train, labels_train, model_path)
        print(train_curve)
