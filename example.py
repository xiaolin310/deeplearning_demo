import torch

import data

# 维数
bptt = 7

batch_size = 4

corpus = data.Corpus("./data")

def batchifx(data, bsz):
    data = data.view(bsz, bptt)
    return data



x_train = batchifx(corpus.train, batch_size)
print(x_train.shape)
y_train = torch.tensor([[0.0,0.0,0.0,1.0], [0.0,0.0,1.0,0.0], [0.0,1.0,0.0,0.0], [1.0,0.0,0.0,0.0]])
model = torch.nn.Sequential(
    torch.nn.Linear(7, 4)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(1000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss;{:.4f}'.format(epoch + 1, 1000, loss.item()))

x_test = torch.tensor([[ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0]])
y_pred = model(x_test)
print('预测结果: ', y_pred.detach().numpy())
