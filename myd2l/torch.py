# /home/d2l/d2l/myd2l/torch.py
#本包用于d2l学习过程中的个人调试与更新，用于重构d2l/torch.py
#还包含/home/d2l/d2l/setup.py &  
# /home/d2l/d2l/pyproject.toml & 
# /home/d2l/d2l/myd2l.egg-info 用于构建本地pkg

#代码更新后,在ipykernel运行时需要重启kernel

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图
    让 Jupyter Notebook 中的 Matplotlib 图形以 SVG（矢量图）格式 显示，而非默认的 PNG。
    backend_inline 是 matplotlib-inline 包提供的模块（Jupyter 内联后端）。
    此调用告诉 Jupyter：“以后所有 matplotlib 图都用 SVG 格式渲染”。
    💡 注意：此设置对当前 notebook 会话全局生效。

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg') 

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小
    plt.rcParams 是 Matplotlib 的全局配置字典。
    'figure.figsize' 控制图形默认宽高（单位：英寸）。

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴
    
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel) # 设置 x 轴标签。
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale) # 设置 x 轴刻度类型（线性/对数等）
    axes.set_yscale(yscale)
    axes.set_xlim(xlim) # 设置 x 轴显示范围（必须是二元 tuple/list）。
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend) # 如果 legend 非空（不是 None 或空列表），则显示图例。
    axes.grid() #显示网格线，便于读图。 


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点
    x 坐标数据（可为 list、tensor、ndarray）
    y 坐标数据；若为 None，则 X 被当作 y 值，x 自动为索引
    fmts: 每条线的样式（颜色+线型）,默认值 ('-', 'm--', ...)

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True， 判断 X 是否是“一维数据”。
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y): # 如果 X 只有一组，但 Y 有多组（常见情况），则复制 X 使其匹配 Y 的数量。
        X = X * len(Y)
    axes.cla() # 清空当前坐标轴，避免旧图残留。
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Timer:
    """记录多次运行时间"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声 造一个“可控的数据集”
    w: 真实权重（tensor，形状 (d,)）
    b: 真实偏置（标量或 0-d tensor）
    num_examples: 样本数

    Defined in :numref:`sec_linear_scratch`"""
    X = torch.normal(0, 1, (num_examples, len(w))) # 每个样本是一个 len(w) 维向量，特征服从标准正态分布
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1)) 

def linreg(X, w, b):
    """线性回归模型

    Defined in :numref:`sec_linear_scratch`"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失
    reshape： 确保y 和 y_hat 形状一致，避免广播错误
    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  

def sgd(params, lr, batch_size):
    """小批量随机梯度下降
    params: [w, b]（tensor 列表）
    lr: 学习率
    batch_size: 批大小

    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad(): # 告诉 autograd：下面的操作不要建计算图
        for param in params:
            param -= lr * param.grad / batch_size # 累积的梯度/ 批大小 = 平均梯度
            param.grad.zero_() #  必须有！PyTorch 默认梯度是 累积的

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器
    torch.utils.data.TensorDataset ：是一个数据集包装器，用于将多个张量打包成一个数据集，要求所有张量的第一维（样本数）相同。
                                    每次索引 dataset[i] 返回 (tensor1[i], tensor2[i], ..., tensorN[i])
                                    常用于：(X, y) 配对，即特征和标签一起迭代
    e.g.: 
    X = torch.randn(100, 2)   # 100 个样本，2 个特征
    y = torch.randn(100, 1)   # 100 个标签

    dataset = TensorDataset(X, y)
    print(dataset[0])  # 输出: (tensor([x1, x2]), tensor([y1]))

    * 是 Python 的 “解包操作符”（unpacking operator） 它把一个可迭代对象（如列表、元组）展开为多个独立参数。

    data_arrays = (features, labels)  # 一个包含两个张量的元组
    # 不使用 *：
    TensorDataset(data_arrays)        # ❌ 错误！传入的是一个元组，不是两个张量

    # 使用 *：
    TensorDataset(*data_arrays)       # ✅ 等价于 TensorDataset(features, labels)
    
    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签

    Defined in :numref:`sec_fashion_mnist`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    #因为我们不需要 fig 对象（不进行保存、调整整体布局等操作）
    #用 _ 是 Python 惯例，表示“忽略这个变量”
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # 将 2D 子图数组转为 1D，便于顺序遍历
    axes = axes.flatten()
    #zip(axes, imgs) 将 axes（子图列表）和 imgs（图像列表）配对,生成迭代器：(axes[0], imgs[0]), (axes[1], imgs[1])
    #enumerate(...) 给每个配对加上索引 i：(0, (ax0, img0)), (1, (ax1, img1)), ...
    for i, (ax, img) in enumerate(zip(axes, imgs)): 
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers(num_process:int)->int:
    """使用4个进程来读取数据

    Defined in :numref:`sec_fashion_mnist`"""
    return num_process

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中
    resize:int|none : 是否将图像调整为指定尺寸（如 resize=32 将 28×28 图像放大到 32×32）
    num_process : 默认为8
    Defined in :numref:`sec_fashion_mnist`"""
    #将 PIL 图像或 NumPy 数组转换为 PyTorch 张量
    #关键行为：将像素值从 [0, 255]（uint8）缩放到 [0.0, 1.0]（float32）
    #改变维度顺序：(H, W, C) → (C, H, W)（通道前置）
    #输出张量形状：对于 Fashion-MNIST（灰度图），变为 (1, 28, 28)
    trans = [transforms.ToTensor()] 
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../Fashion-MNIST_data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../Fashion-MNIST_data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers(num_process=8)),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers(num_process=8)))

def accuracy(y_hat, y):
    """计算预测正确的数量

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.to(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度

    Defined in :numref:`sec_softmax_scratch`"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）

    Defined in :numref:`sec_softmax_scratch`"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """用于后面lambda闭包
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(16, 9)):
        """Defined in :numref:`sec_softmax_scratch`"""

        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        #如果只有 1 个子图，axes 是单个对象；但为了统一处理，强制转为列表 [axes]
        #这样后续代码总能用 self.axes[0] 访问第一个（也是唯一一个）坐标轴
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        # 支持标量或列表
        #如果 y 是单个数字（如 loss=0.5），转为 [0.5]
        #如果 x 是单个数字（如 epoch=3），复制成 [3, 3, ..., 3]（长度 = y 的条数）
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        #遍历每条线，将 (x, y) 追加到历史记录中
        #跳过 None 值（可用于跳过某些 epoch 不画点）
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla() # clear current axes
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        #清除上一次输出
        #wait=True 表示“等新内容准备好再清除”，避免闪烁
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）

    Defined in :numref:`sec_softmax_scratch`"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


from .utils import argmax, reshape
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）
    预测标签并可视化前 n 个样本（Fashion-MNIST 专用）
    
    参数:
        net: 模型函数，接受 (batch_size, 784) 输入，输出 (batch_size, 10) logits 或概率
        test_iter: 测试数据迭代器（DataLoader），每个 batch 为 (X, y)
        n: 显示前 n 张图像（默认 6）
    Defined in :numref:`sec_softmax_scratch`"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(torch.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])
