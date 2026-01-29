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