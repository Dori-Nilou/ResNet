import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets, models
#from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
#import torch.optim as optim
from torch import nn

from app.resnet import *
from app.hparams import *

g_model = None
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def save_params(filename):
    global g_model, device
    torch.save(g_model.state_dict(), filename)


def train():
    global g_model

    train_batch_size = hparams.BATCH_SIZE
    num_epoches = hparams.NUM_EPOCH
    lr = hparams.LR
    optim_type = hparams.OPTIMIZER
    input_size = hparams.INPUT_SIZE

    # load datasets
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    path = './app/datasets/WHU-RS19-train-7'
    train_data = datasets.ImageFolder(path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    print('load WHU-RS19: success\n')

    criterion = nn.CrossEntropyLoss()
    if(optim_type == "Adam"):
        optimizer = torch.optim.Adam(g_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif optim_type == "SGD":
        optimizer = torch.optim.SGD(g_model.parameters(), lr = lr, weight_decay=1e-5)

    losses = []
    acces = []

    for epoch in range(num_epoches):
        train_loss = 0
        train_acc1 = 0
        train_acc5 = 0
        g_model.train()
        if (epoch+1)%5 == 0 :
                optimizer.param_groups[0]['lr'] *= 0.1
                print('lr: {}'.format(optimizer.param_groups[0]['lr']))
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            out = g_model(img)

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            train_loss += loss.item() # 整个训练集上的累计loss

            # top1 acc
            _, pred = out.max(1) # pred为out中每个样本输出概率中最大的值的索引，也就是其对应的类别
            num_correct = (pred == label).sum().item() # 求预测正确的样本个数
            acc = num_correct / img.shape[0] # 单个batch上的准确率
            train_acc1 += acc # 整个训练集上的累计准确率（top1）

            maxk = max((1, 5))
            _, pred1 = out.topk(maxk, 1, True, True)
            label_resize = label.view(-1, 1)
            train_acc5 += torch.eq(pred1, label_resize).sum().float().item() / img.shape[0] 

        # len(train_loader)为batch的数量，这里求每个batch的平均loss和acc
        print('epoch: {}, Train Loss: {:.4f}, Train Acc-top1: {:.4f}, Train Acc-top5: {:.4f}'
                .format(epoch, train_loss / len(train_loader), train_acc1 / len(train_loader),\
                        train_acc5 / len(train_loader)))


def inference():
    global g_model
    g_model.eval()

    test_batch_size = hparams.BATCH_SIZE
    input_size = hparams.INPUT_SIZE

    # load datasets
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    path = './app/datasets/WHU-RS19-test-3'
    test_data = datasets.ImageFolder(path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    test_acc1 = 0
    test_acc5 = 0

    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)

        out = g_model(img)

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        test_acc1 += num_correct / float(img.shape[0])

        # top5 acc
        maxk = max((1, 5))
        _, pred1 = out.topk(maxk, 1, True, True)
        label_resize = label.view(-1, 1)
        test_acc5 += torch.eq(pred1, label_resize).sum().float().item() / img.shape[0] 

    print("Inference Acc-top1: {:.4f}, Acc-top5: {:.4f}"
          .format(test_acc1 / len(test_loader), test_acc5 / len(test_loader)))


def main():   
    global g_model, device
    hparams.load_json('default.json')

    file_path = './model/resnet/'
    filename = file_path + str(hparams.NET_TYPE) + '_' + str(hparams.OPTIMIZER) + '_NE' + \
        str(hparams.NUM_EPOCH) + '_B' + str(hparams.BATCH_SIZE) + '_inSIZE_' + str(hparams.INPUT_SIZE) +\
              '_ID_' + str(hparams.ID) + '.pth'

    if hparams.NET_TYPE == "resnet20":
        g_model = resnet20()
    elif hparams.NET_TYPE == "resnet32":
        g_model = resnet32()
    elif hparams.NET_TYPE == "pretrained":
        g_model = models.resnet34(pretrained=False)
        g_model.load_state_dict(torch.load("./model/resnet/resnet34-333f7ec4.pth"))
        num_ftrs = g_model.fc.in_features
        g_model.fc = nn.Linear(num_ftrs, 19)
        
    g_model = g_model.to(device=device)

    if(hparams.MODE == "train"):
        train()
        save_params(filename=filename)
        print("save params to" + filename + ": Done\n")
    elif(hparams.MODE == "inference"):
        g_model.load_state_dict(torch.load(filename))
        print("load params from" + filename + ": Done\n")
        inference()


if __name__ == '__main__':
    main()
