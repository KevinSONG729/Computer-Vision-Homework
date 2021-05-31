import cv2 as cv
import os
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		# 卷积层
		self.conv = torch.nn.Sequential(
			# 输入是1，输出是16，卷积核是3*3，每次移动1步，每一条边补充1行/列的0，
			# 经过这一步之后，数据由1*28*28，变成了16*28*28
			torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			# 激活函数
			torch.nn.ReLU(),
			# 输入是16，输出是32，卷积核是3*3，每次移动1步，每一条边补充1行/列的0，
			# 经过这一步之后，数据由16*28*28，变成了32*28*28
			torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			# 激活函数
			torch.nn.ReLU(),
			# 池化层
			# 每次移动2步，卷积核是2*2，即每2*2的矩阵中的4个数据选取一个最大值，
			# 这样就由32*28*28，变成了32*14*14
			torch.nn.MaxPool2d(stride=2, kernel_size=2)
		)

		# 全连接层
		self.dense = torch.nn.Sequential(
			# 将32*14*14的数据线性转化为1024的数据
			torch.nn.Linear(32 * 25 * 25, 1024),
			# 激活函数
			torch.nn.ReLU(),
			# 将1024的数据线性转化为10的数据，即0-9
			torch.nn.Linear(1024, 10)
		)

	# 向前传播
	def forward(self, x):
		# 卷积
		x = self.conv(x)
		# 扁平化处理
		x = x.view(-1, 32 * 25 * 25)
		# 全连接
		x = self.dense(x)
		return x

if __name__ == "__main__":
    print("detect object")