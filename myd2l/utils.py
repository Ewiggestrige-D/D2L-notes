# myd2l/torch_utils.py
import torch
import matplotlib.pyplot as plt

def argmax(X, axis=None):
    return torch.argmax(X, dim=axis)

def reshape(X, shape):
    return X.reshape(shape)

