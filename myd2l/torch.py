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