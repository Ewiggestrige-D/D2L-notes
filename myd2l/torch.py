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

"""
凡是：

不参与前向 / 反向传播

不进入训练循环 hot path

只是 I/O、工具、设备选择

👉 不需要本地化，直接用即可
"""

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
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(8, 4.5), axes=None):
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
                 figsize=(8, 4.5)):
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
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    print(f'train_loss ={train_metrics[0]},\n train_accuracy ={train_metrics[1]},\n test_accuracy={test_acc}')

# from .utils import argmax, reshape
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


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失

    Defined in :numref:`sec_model_selection`"""
    metric = Accumulator(2) # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(torch.sum(l), l.numel())
    return metric[0] / metric[1]

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件

    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件

    Defined in :numref:`sec_kaggle_house`"""
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
def corr2d(X, K):
    """
    计算二维互相关（cross-correlation）

    参数:
        X: 输入张量，shape = (H, W)
        K: 卷积核，shape = (h, w)

    返回:
        Y: 输出张量，shape = (H-h+1, W-w+1)

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    # 1. 创建输出张量（全 0）
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1),
                    dtype=X.dtype,
                    device=X.device
                    )
    # 2. 滑动窗口计算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = torch.sum((X[i: i + h, j: j + w] * K))
    return Y

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(
        xlabel='epoch', 
        xlim=[1, num_epochs],
        legend=['train loss', 'train acc', 'test acc']
        )
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        """Defined in :numref:`sec_hybridize`"""
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

DATA_HUB['time_machine'] = (
    DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
    )

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中

    Defined in :numref:`sec_text_preprocessing`"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元

    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        tokens: 二维或一维 token 列表（如 [['a','b'], ['c']]）
        min_freq: 最小出现频次，低于此值的 token 被忽略
        reserved_tokens: 预留 token（如 ['<pad>', '<bos>']），总是包含在词表开头

        内部结构：
        idx_to_token: 列表，索引 → token（如 [0:'<unk>', 1:'a', 2:'b', ...]）
        token_to_idx: 字典，token → 索引（如 {'a':1, 'b':2, ...}）
        _token_freqs: 按频次降序排列的 (token, freq) 列表
        
        特殊设计：
        <unk> 固定索引为 0：任何未登录词（OOV）都映射到 0
        预留 token 优先插入：确保 <pad> 等有固定 ID
        
        Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """功能：token → index 的便捷接口
        支持两种调用：
        vocab['hello'] → 返回 ID（若不存在，返回 0）
        vocab[['h','e','l']] → 返回 [ID_h, ID_e, ID_l]
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        功能：index → token 的反向映射
        用途：将模型输出的 ID 序列转回可读文本
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率

    Defined in :numref:`sec_text_preprocessing`"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表

    Defined in :numref:`sec_text_preprocessing`"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列

    从一个长序列 corpus（如 [1,2,3,4,5,6,7,8,...]）中，随机抽取不重叠的子序列，每个子序列长度为 num_steps，并构造：
    输入 X：子序列的前 num_steps 个 token（历史上下文）
    标签 Y：子序列的后 num_steps 个 token（即 X 向右平移 1 位）
    
    Defined in :numref:`sec_language_model`"""

    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    # 避免所有 batch 总是从序列的“整数倍位置”开始（如 0, L, 2L...），增加随机性。
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    # 每个子序列需要 num_steps 个输入 + num_steps 个标签，但标签是输入的“下一个”
    # 总共需要 num_steps + 1 个连续 token！
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # 将列表转为 PyTorch 张量
        # X 和 Y 的形状均为 (batch_size, num_steps)
        # 使用 yield 实现内存高效的迭代器
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列
    它特别适用于需要维持序列连续性的场景，比如 RNN 训练中希望跨 batch 传递隐藏状态。

    从长序列 corpus 中，按顺序、不打乱地划分出多个 batch，每个 batch 包含 batch_size 条长度为 num_steps 的子序列，并保证：
    - 同一批内：不同样本是原始序列中等间隔的片段（用于并行）
    - 批与批之间：在原始序列中连续衔接（可用于状态传递）
    ✅ 这是 stateful RNN 训练的标准方式。
    """
    # 从随机偏移量开始划分序列
    # 随机起始偏移（避免固定对齐）
    offset = random.randint(0, num_steps)
    # 我们需要满足两个条件：
    # - X 和 Y 都要有足够长度 → 需要 num_tokens + 1 个原始 token（Y 比 X 多一个位置）
    # - 总 token 数必须能被 batch_size 整除 → 以便 reshape 成 (batch_size, T)
    # 所以：
    # 可用长度 = len(corpus) - offset（去掉开头偏移）
    # 但 Y 需要再往后一个位置 → 最大可用 = len(corpus) - offset - 1
    # 然后向下取整到 batch_size 的倍数：// batch_size * batch_size
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    # 构造完整的 Xs 和 Ys（全局视图）
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    # 重塑为 (batch_size, T) —— “垂直切分” 
    # 将长序列“竖着”切成 batch_size 条平行轨道
    """
    Xs = [
    [x1, x4, x7, x10],   # 轨道 0
    [x2, x5, x8, x11],   # 轨道 1
    [x3, x6, x9, x12]    # 轨道 2
    ]
    ✅ 每条轨道是原始序列中每隔 batch_size 个位置取一个 token
    ✅ 轨道之间在时间上是交错的，但各自保持连续

    为什么这样做？
    - 允许在一个 batch 中并行处理 batch_size 个独立但连续的子序列
    - 对于 RNN，每条轨道可以维护自己的隐藏状态
    """
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    # 逐 batch 生成（滑动窗口） 
    # 沿时间维度滑动窗口，每次取 num_steps 列
    # X 和 Y 形状均为 (batch_size, num_steps)
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表

    Defined in :numref:`sec_language_model`"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """Defined in :numref:`sec_rnn_scratch`"""
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符

    Defined in :numref:`sec_rnn_scratch`"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

class RNNModel(nn.Module):
    """循环神经网络模型

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

#d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
#                          '94646ad1522d915e7b0f9296181140edcf86a4f5')