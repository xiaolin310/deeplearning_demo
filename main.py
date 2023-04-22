import torch

hard_features_map = {
    "男": [1.0, 0.0],
    "女": [0.0, 1.0],
    "23": [1.0, 0.0, 0.0, 0.0],
    "30": [0.0, 1.0, 0.0, 0.0],
    "35": [0.0, 0.0, 1.0, 0.0],
    "45": [0.0, 0.0, 0.0, 1.0],
    "中专": [1.0, 0.0, 0.0, 0.0],
    "大专": [0.0, 1.0, 0.0, 0.0],
    "本科": [0.0, 0.0, 1.0, 0.0],
    "硕士": [0.0, 0.0, 0.0, 1.0],
    "计算机": [1.0, 0.0, 0.0, 0.0],
    "物流管理": [0.0, 1.0, 0.0, 0.0],
    "市场营销": [0.0, 0.0, 1.0, 0.0],
    "临床医学": [0.0, 0.0, 0.0, 1.0],
    "1年": [1.0, 0.0, 0.0, 0.0],
    "3年": [0.0, 1.0, 0.0, 0.0],
    "5年": [0.0, 0.0, 1.0, 0.0],
    "7年": [0.0, 0.0, 0.0, 1.0],
    "苏州": [1.0, 0.0, 0.0, 0.0],
    "天津": [0.0, 1.0, 0.0, 0.0],
    "杭州": [0.0, 0.0, 1.0, 0.0],
    "武汉": [0.0, 0.0, 0.0, 1.0],
    "3000": [1.0, 0.0, 0.0, 0.0],
    "5000": [0.0, 1.0, 0.0, 0.0],
    "10000": [0.0, 0.0, 1.0, 0.0],
    "15000": [0.0, 0.0, 0.0, 1.0],
    "软件工程师": [1.0, 0.0, 0.0, 0.0],
    "快递员": [0.0, 1.0, 0.0, 0.0],
    "销售": [0.0, 0.0, 1.0, 0.0],
    "医生": [0.0, 0.0, 0.0, 1.0]
}

# batch size
m = 50
# features 26维
n = 26
# labels 4维
p = 4

epoches = 100000
features = list()
x_train = list()

labels = list()

i = 0
with open("./data/train.txt") as f:
    for line in f:
        for word in line.strip().split(",")[:-1]:
            # print(hard_features_map[word])
            x_train.extend(hard_features_map[word])
            # print(x_train)
        # print(len(x_train))
        if i == 10:
            print(x_train)
        features.append(x_train)
        x_train = []
        i += 1

# print(features)

with open("./data/train.txt") as f:
    for line in f:
        labels.append(hard_features_map[line.strip().split(",")[-1]])


# print(labels)

x_train = torch.Tensor(features)
y_train = torch.Tensor(labels)

model = torch.nn.Sequential(
    torch.nn.Linear(26, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 4),
    torch.nn.Softmax(dim=1)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(epoches):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print('Epoch [{}/{}], Loss;{:.4f}'.format(epoch + 1, epoches, loss.item()))

x_test = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
with torch.no_grad():
    y_pred = model(x_test)
    print('预测结果: ', y_pred.detach().numpy())
