import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

batch_size = 100
# MNIST DATASET
train_dataset = dsets.MNIST(root = '/pymnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root = '/pymnist', train=False, transform=transforms.ToTensor(), download=True)
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

input_size = 784
hidden_size = 500
num_classes = 10

class Simple_Net(nn.Module):
    def __init__(self, input_num, hidden_size, out_put):
        super(Simple_Net, self).__init__()
        self.layer1 = nn.Linear(input_num, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_put)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.relu(out)
        out = self.layer2(out)
        return out
net = Simple_Net(input_size, hidden_size, num_classes).cuda()
print(net)

# 训练
learning_rate = 1e-1
num_epoches = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
for epoch in range(num_epoches):
    print("current epoch is " + str(epoch))
    for i, (image, labels) in enumerate(train_loader):
        image = Variable(image.view(-1 , 784)).cuda()
        labels = Variable(labels).cuda()

        outputs = net(image)
        loss = criterion(outputs, labels).cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i % 100 == 0):
            print("current loss is " + str(loss.item()))
print("finished training")

# 预测
total = 0
correct = 0

for images, labels in test_loader:
    images = Variable(images.view(-1,784)).cuda()
    outputs = net(images)

    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()

print("Accuracy = " + str(float(100 * correct / total)))


