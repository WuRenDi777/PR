# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:20:24 2023

@author: RONGFU LI
"""



import numpy as np
import os
from scipy.io import loadmat
from scipy.stats import zscore
from numpy.lib.stride_tricks import sliding_window_view

#%%  注意：用的是DB2中第一个人的第一类手势数据验证




def load_data(filename):
    # 加载.mat文件
    data = loadmat(filename)
    emg = data['emg']
    stimulus = data['stimulus']
    return emg, stimulus

# 读取你的所有数据,电脑多了带不动，加上你需要的数据就可以了，注意：这次选择的是前五个数据合一起作为训练和测试集的
# filenames = ['S1_E1_A1.mat', 'S2_E1_A1.mat', 'S3_E1_A1.mat', 'S4_E1_A1.mat', 'S5_E1_A1.mat']

filenames = ['S1_E1_A1.mat']

Combined = []
for name in filenames:
    emg, stimulus = load_data(name)
    # 将肌电信号和标签合并，并只保留标签不为0的数据
    combined = np.hstack((emg, stimulus.reshape(-1,1))) #将stimulus变成列向量并与emg拼接
    combined = combined[combined[:,-1] != 0]
    Combined.append(combined)
    
    

    
#%%
    
def split_data_by_gesture(combined, num_repeats):
    # 获取所有的手势标签
    gestures = np.unique(combined[:, -1])

    train_data = []
    test_data = []
    for gesture in gestures:
        # 对每种手势，将数据分为6份，每份对应一个重复
        gesture_data = combined[combined[:, -1] == gesture]
        data_splits = np.array_split(gesture_data, num_repeats)
        
        # 划分训练集和测试集
        for i, data in enumerate(data_splits, start=1):
            if i in [1, 3, 4, 6]:  # 1、3、4、6号重复产生的样本划分为训练集
                train_data.append(data)
            else:  # 2、5号重复产生的样本划分为测试集
                test_data.append(data)
    
    # 合并训练集和测试集
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    
    return train_data, test_data

# 设置重复次数
num_repeats = 6

# 对每个combined进行划分
train_data = []
test_data = []
for combined in Combined:
    train, test = split_data_by_gesture(combined, num_repeats)
    train_data.append(train)
    test_data.append(test)

#%%


def preprocess_data(emg, labels, w, s, f):
    window_size = int(w * f / 1000)  # 根据公式计算窗口大小
    step_size = int(s * f / 1000)  # 计算步长

    # 初始化存储预处理后数据和标签的列表
    preprocessed = []
    preprocessed_labels = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        # 对每个标签，将数据分为窗口，每个窗口对应一个标签
        emg_label = emg[labels == label]
        for i in range(0, len(emg_label) - window_size, step_size):
            window = emg_label[i:i + window_size]
            preprocessed.append(zscore(window))  # 对窗口数据进行标准化并存储
            preprocessed_labels.append(label)  # 保存对应的标签
    
    return np.array(preprocessed), np.array(preprocessed_labels)

# 对训练集和测试集进行预处理
processed_train_data = []
processed_train_labels = []
processed_test_data = []
processed_test_labels = []



# 设置滑动窗口参数
f = 2000  # 采样频率为2000Hz
w = 100  # 窗口长度，单位毫秒
s = 100  # 滑动步长，单位毫秒


for train, test in zip(train_data, test_data):
    processed_train, labels_train = preprocess_data(train[:, :-1], train[:, -1], w, s, f)  # 注意我们排除了标签列
    processed_test, labels_test = preprocess_data(test[:, :-1], test[:, -1], w, s, f)  # 注意我们排除了标签列
    
    processed_train_data.append(processed_train)
    processed_train_labels.append(labels_train)
    processed_test_data.append(processed_test)
    processed_test_labels.append(labels_test)

# 合并所有的训练数据和测试数据
processed_train_data = np.concatenate(processed_train_data)
processed_test_data = np.concatenate(processed_test_data)
train_labels = np.concatenate(processed_train_labels)
test_labels = np.concatenate(processed_test_labels)

#常规图片形式
processed_train_data = processed_train_data.transpose((0, 2, 1))  # 转换为 (batch_size, num_features, sequence_length) 的形状
processed_test_data = processed_test_data.transpose((0, 2, 1))  # 转换为 (batch_size, num_features, sequence_length) 的形状


# 最后，可以使用 processed_train_data, train_labels, processed_test_data 和 test_labels 进行模型训练和测试。

#%%


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 设定GPU运算如果可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设定超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 150#可以改
num_classes = 17  # 根据你的数据类别数量进行修改

# 数据处理，转换为torch能接受的tensor形式
train_data = torch.tensor(processed_train_data).float().to(device)
train_labels = torch.tensor(train_labels).long().to(device)
test_data = torch.tensor(processed_test_data).float().to(device)
test_labels = torch.tensor(test_labels).long().to(device)

# 标签修正，将标签从1-17调整为0-16
train_labels -= 1
test_labels -= 1

# 封装为dataloader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#%% 加LSTM
# 设定模型
class ChannelConvModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelConvModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=20, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=10, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, padding=1)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=4, padding=1)
        self.pool5 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 64, batch_first=True)  # 添加LSTM层

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.transpose(1, 2)  # 交换 num_features 和 sequence_length 的维度
        x, _ = self.lstm(x)
        return x[:, -1, :]  # 只使用最后一个时间步的输出

class MyModel(nn.Module):
    def __init__(self, num_channels=12):
        super(MyModel, self).__init__()
        self.channel_convs = nn.ModuleList([ChannelConvModule(3) for _ in range(num_channels // 3)])  # 每个ChannelConvModule处理3个通道
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64 * (num_channels // 3), 128)  # 调整全连接层的输入大小
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        conv_results = []
        for i, conv in enumerate(self.channel_convs):
            result = conv(x[:, 3*i:3*(i+1), :])  # 将每3个通道的数据一起送入同一个ChannelConvModule
            conv_results.append(result)
        x = torch.cat(conv_results, dim=-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel().to(device)

#%%不加LSTM


# 设定模型
class ChannelConvModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelConvModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=20, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=10, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, padding=1)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=4, padding=1)
        self.pool5 = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.gap(x)  # 应用全局平均池化
        return x.squeeze()  # 删除单维度条目，形状为(batch_size, 64)

class MyModel(nn.Module):
    def __init__(self, num_channels=12):
        super(MyModel, self).__init__()
        self.channel_convs = nn.ModuleList([ChannelConvModule(3) for _ in range(num_channels // 3)])  # 每个ChannelConvModule处理3个通道
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64 * (num_channels // 3), 128)  # 调整全连接层的输入大小
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        conv_results = []
        for i, conv in enumerate(self.channel_convs):
            result = conv(x[:, 3*i:3*(i+1), :])  # 将每3个通道的数据一起送入同一个ChannelConvModule
            conv_results.append(result)
        x = torch.cat(conv_results, dim=-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel().to(device)

#%%12个通道分别进行五个卷积层，在合并


# 设定模型
# 定义一个卷积模块，用于处理一个通道的数据
class ChannelConvModule(nn.Module):
    def __init__(self):
        super(ChannelConvModule, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)  # 第一层卷积，输出通道数为8
        self.pool1 = nn.MaxPool1d(2)  # 第一层最大池化
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)  # 第二层卷积，输出通道数为16
        self.pool2 = nn.MaxPool1d(2)  # 第二层最大池化
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)  # 第三层卷积，输出通道数为32
        self.pool3 = nn.MaxPool1d(2)  # 第三层最大池化
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1)  # 第四层卷积，输出通道数为32
        self.pool4 = nn.MaxPool1d(2)  # 第四层最大池化
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # 第五层卷积，输出通道数为64
        self.pool5 = nn.MaxPool1d(2)  # 第五层最大池化
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层

    def forward(self, x):
        # 输入经过五个卷积层和最大池化层，然后通过全局平均池化层
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.gap(x)  # 应用全局平均池化
        return x

# 主模型
class MyModel(nn.Module):
    def __init__(self, num_channels=12):
        super(MyModel, self).__init__()
        # 对每个通道，都创建一个 ChannelConvModule
        self.channel_convs = nn.ModuleList([ChannelConvModule() for _ in range(num_channels)])
        self.dropout = nn.Dropout(p=0.3)  # Dropout层
        # 全连接层，注意输入大小应为“每个通道的特征数（这里是64，由最后一层卷积决定）乘以通道数”
        self.fc = nn.Linear(64 * num_channels, num_classes)  

    def forward(self, x):
        # 输入形状应为 (batch_size, num_channels, sequence_length)
        # 将输入转置为 (batch_size, sequence_length, num_channels)
        x = x.transpose(1, 2)  
        conv_results = []
        # 遍历每个通道，通过卷积模块处理，并将结果收集到列表中
        for i, conv in enumerate(self.channel_convs):
            result = conv(x[:, :, i].unsqueeze(1))  # 添加通道维度，并通过卷积模块
            result = result.view(result.size(0), -1)  # 展平输出
            conv_results.append(result)
        x = torch.cat(conv_results, dim=-1)  # 沿通道维度拼接结果
        x = self.dropout(x)  # 通过Dropout层
        x = self.fc(x)  # 通过全连接层
        return x

model = MyModel().to(device)








#%%

# 设定损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}')
















    