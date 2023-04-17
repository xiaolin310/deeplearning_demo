import torch
import torch.nn as nn
import torch.optim as optim

import data

# 输入shape
m = 100
n = 8

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 64)  # 输入层到隐层1
        self.fc2 = nn.Linear(64, 128)  # 隐层1到隐层2
        self.fc3 = nn.Linear(128, 1)   # 隐层2到输出层
        self.relu = nn.ReLU()           # ReLU激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))      # 第一层全连接 + 激活
        x = self.relu(self.fc2(x))      # 第二层全连接 + 激活
        x = self.fc3(x)                 # 输出层全连接
        return x

def batchify(data):
    return data.view(m, n)

corpus = data.Corpus("./data")

# 分离训练，特征和标签
x, y = torch.split(batchify(corpus.train), [7, 1], dim=1)
y = y.view(-1)
print(x)
print(y)



# 加载数据集
train_set = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

# 初始化模型和损失函数
net = Net()
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降进行优化
optimizer = optim.SGD(net.parameters(), lr=0.02)

# 训练模型
for epoch in range(20): # 进行 10 次迭代
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad() # 梯度清零
        outputs = net(inputs) # 前向传播
        loss = criterion(outputs.view(-1), labels) # 计算损失
        loss.backward()       # 反向传播
        optimizer.step()      # 更新参数
        running_loss += loss.item()
        if i % 9 == 0:    # 每 10 批次打印一次损失值
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0

print('Finished Training')

# # 推理测试
# test_inputs = torch.tensor([[0., 2., 4., 6., 8., 10., 12.]])
# with torch.no_grad():
#     test_outputs = net(test_inputs)
# _, predicted =
