# ResNet

## 环境
pytorch 1.9.0
python 3.7.16

## 入口程序
train_resnet.py

## 超参数设置
default.json

## 数据集
链接：https://captain-whu.github.io/BED4RS/#

将数据集根目录命名为WHU-RS19，放置在./app/datasets目录下，在该目录下运行split_dataset.py划分数据集

## PS：
貌似还是有点问题，使用torchvision.models中预训练的模型可以较快收敛到高准确率，但使用./app/resnet.py中的模型仍然准确率低
