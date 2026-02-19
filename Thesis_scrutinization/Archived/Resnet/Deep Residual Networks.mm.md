---
title: Identity Mappings in Deep Residual Networks
markmap:
  colorFreezeLevel: 12
---

## Identity Mappings in Deep Residual Networks

- [Local Thesis](../Resnet/1603.05027v3_Identity%20Mappings%20in%20Deep%20Residual%20Networks.pdf)
- [Code](https://github.com/KaimingHe/resnet-1k-layers)
- Basic info 
  - Author:
    - Kaiming He*（Main）
    - Xiangyu Zhang 
    - Shaoqing Ren
    - Jian Sun
  - Publish:
    - Time: 2016
    - Journal: ECCV

## Abstract
- **Idea**
    - the forward and backward signals can be directly propagated from one block to any other block when using **identity mappings** as skip connections and after-addition activation.

## 1 Introduction
- $$x_{l+1} = ReLU \left(h(x_{l})+\mathcal{F} (x_{l}, \mathcal{W}_{l})\right)$$
    $$x_{l+1} = f(y_{l})$$
    $h(x_{l})= x_{l}$ : identity mapping

- conclusions :
    1. training  becomes easier when $h(x_{l})$ and
    $f(y_{l})$ are **identity mappings**
    2. identity mapping $h(x_{l})$ achieves *the fastest error reduction and lowest training loss*
    3. **a clean information path** is helpful for easing optimization.
    4. **new residual unit**:  **ReLU and BN**  as *pre-activation* of the weight layers

## 2 Analysis of Deep Residual Networks
- Formula: $$x_{L}=x_{l} + \sum_{i=l}^{L-1} \mathcal{F} (x_{i}, \mathcal{W}_{i})$$ 
    *for any deeper unit $L$ and any shallower unit $l$*
    - property
        - The feature $x_{L}$ of any deeper unit $L$ can be
        any shallower feature $x_{l}$ plus a residual function 
        - $x_{L}=x_{0} + \sum_{i=0}^{L-1} \mathcal{F} (x_{i}, \mathcal{W}_{i})$
        The feature $x_{L}$ is the summation of the outputs of all preceding residual functions plus $x_{0} $
    - Back-Propagation
        - $$\frac{\partial \mathcal{L}}{\partial x_{l}} =\frac{\partial \mathcal{L}}{\partial x_{L}}\frac{\partial x_{L}}{\partial x_{l}} =\frac{\partial \mathcal{L}}{\partial x_{L}} \left(1+\frac{\partial}{\partial x_{l}} \sum_{i=l}^{L-1} \mathcal{F} (x_{i}, \mathcal{W}_{i}) \right)$$
            - $\frac{\partial \mathcal{L}}{\partial x_{L}} $ propagates information *directly without concerning any weight layers*,
                - ensures that information is directly propagated back to any shallower unit $l$.
            - $\frac{\partial \mathcal{L}}{\partial x_{L}} \frac{\partial}{\partial x_{l}} \sum_{i=l}^{L-1} \mathcal{F} (x_{i}, \mathcal{W}_{i})$ propagates *through the weight layers*
        - gradient of a layer **does not vanish** even when the weights are arbitrarily **small**. 
            - $ \frac{\partial}{\partial x_{l}} \sum_{i=l}^{L-1} \mathcal{F} (x_{i}, \mathcal{W}_{i})$ cannot be always **-1** for all samples in a mini-batch.
- Disscusion:signal can be directly propagated from any unit to another, ***both forward and backward***.


## 3 On the Importance of Identity Skip Connections

### Simple Modification
- modification : $h(x_{l}) =\lambda_{l} x_{l}$ 
    - Forward-Propagation
        - $$x_{L}=(\prod_{i=l}^{L-1}\lambda_{i})x_{l} + \sum_{i=l}^{L-1} (\prod_{j=i+1}^{L-1}\lambda_{j})\mathcal{F} (x_{i}, \mathcal{W}_{i})$$
    - Backward-Propagation
        - $$\frac{\partial \mathcal{L}}{\partial x_{l}} =\frac{\partial \mathcal{L}}{\partial x_{L}} \left((\prod_{i=l}^{L-1}\lambda_{i})+\frac{\partial}{\partial x_{l}} \sum_{i=l}^{L-1} \hat{\mathcal{F}} (x_{i}, \mathcal{W}_{i}) \right)$$
            - $(\prod_{i=l}^{L-1}\lambda_{i})$ leads to **gradient vanish/explode**

### 3.1 Experiments on Skip Connections
- baseline ResNet-110  error rate：6.61% 
- Constant scaling
    - $\lambda =0.5$
        - $\mathcal{F}$ is not scaled
            - **not converge!!**
        - $\mathcal{F}$ scaled by $1-\lambda$
            - error rate：12.35%  (much higher)
- **Exclusive** gating
    - $\mathcal{F}$ scaled by $g(x) = \text{Sigmoid}(W_{g}x+b_{g})$
    $h(x_{l})$ scaled by $1- g(x)$ 
      - initialization of the biases $b_{g}$ is critical
      - with hyper-parameter search on the initial value of $b_{g}$
      - error rate：8.70% (still lagging) 
        - **Two-Fold Note**:
            - when $1- g(x) \rightarrow 1$, the **gated shortcut connections** are closer to identity which helps
            information propagation; 
            but in this case $g(x) \rightarrow 0$, and suppresses the residul function.
- **Shortcut-only** gating
    - $h(x_{l})$ scaled by $1- g(x)$ 
    $\mathcal{F}$ **unchanged**
      -  initialization of the biases $b_{g}$ is critical
        - $b_{g} = 0$
            - error rate：12.86% (worse) 
        - $b_{g} = -6$
            - error rate：6.91% (**close to baseline**) 

- $1 \times 1$  convolutional shortcut
    - ResNet-34 (16 Residual Units)
        - good results
    - ResNet-110 (54 Residual Units)
        - error rate：12.22% (worse) 
    - **Note** :When stacking so many Residual Units , 
    even the *shortest path* may still impede signal propagation. 

- dropout shorcut
    - ratio: 0.5
    - fropout on the output of the identity shortcut
    - result： **fails to converge **
        - Dropout statistically imposes a scale of $\lambda$
        with an expectation of 0.5 on the shortcut, thus impedes signal propagation.

### 3.2 Discussions
- conlusions
    - shortcut connections are the **most direct** paths for the information to propagate
    - Multiplicative manipulations on the shortcuts can **hamper**
    information propagation and lead to **optimization problems**
    - gating and convolutional shortcuts introduce **more parameters**,
    but no more optimization
    - degradation of these models is caused by **optimization issues**, instead of representational abilities.

## 4 On the Usage of Activation Functions

### original (Fig4.a)
- Branch 1: weight params + BN + ReLU + weight params + BN
- Branch 2: indentity mapping
- Branch 1 + Branch 2 (addition)
- ReLU
- Test set: CIFAR-10
- Result
    - ResNet-110
        - Classification error rate: 6.61%
    - ResNet-164
        - Classification error rate: 5.93%

### 4.1 Experiments on Activation
- BN after addition (Fig4.b)
    - Architecture
        - Branch 1: weight params + BN + ReLU + weight params 
        - Branch 2: indentity mapping
        - Branch 1 + Branch 2 (addition)
        - BN + ReLU
    - Result： **Much Worse**
        - ResNet-110
            - Classification error rate: 8.17%
        - ResNet-164
            - Classification error rate: 6.50%
    - *Note*
        - BN impedes clear shortcut information propagation
- ReLU before addition (Fig4.c)
    - Architecture
        - Branch 1: (weight params + BN + ReLU ) $\times 2$
        - Branch 2: indentity mapping
        - Branch 1 + Branch 2 (addition)
    - Result： **slightly Worse**
        - ResNet-110
            - Classification error rate: 7.84%
        - ResNet-164
            - Classification error rate: 6.14%
    - *Note*
        - ReLU leads to a non-negative output on **residual** 
- ReLU-only pre-activation (Fig4.d)
    - Architecture
        - Branch 1: (**ReLU** + weight params + BN ) $\times 2$
        - Branch 2: indentity mapping
        - Branch 1 + Branch 2 (addition)
    - Result： **Equal Baseline**
        - ResNet-110
            - Classification error rate: 6.71%
        - ResNet-164
            - Classification error rate: 5.91%
    - *Note*
        - Using asymmetric **after-addition activation** is equivalent to constructing a **pre-activation Residual Unit**
        - the **position** of activation matters (post-activation/pre-activation)
        - 
- Full pre-activation (Fig4.e)
    - Architecture
        - Branch 1: (**BN + ReLU** + weight params) $\times 2$
        - Branch 2: indentity mapping
        - Branch 1 + Branch 2 (addition)
    - Result： **Better and Best**
        - ResNet-110
            - Classification error rate: 6.37%
        - ResNet-164
            - Classification error rate: 5.46%
    - *Note*:
        - *pre-activation* models are better than the baseline **in general** (Table 3)

### 4.2 Analysis: why *pre-activation* better
- **Ease of optimization**
    - $f = $ ReLU,  the signal is impacted if it is negative and truncation when deep net
    - $f = $ ReLU not severe when the ResNet has fewer layers
    - $f = $ identity mapping, signal can be propagated directly and **reduce training loss**

- **Reducing overfitting**
    - in our *pre-activation*, the inputs to all weight layers have been normalized

## 5 Results
- Comparisons on CIFAR-10/100 : 
    - ResNet-1001
    - Params: 10.2M
    - error rate (Best)
        - 4.62% on CIFAR-10
        - 22.71 on CIFAR-100
- Comparisons on ImageNet
    - pre-actiovation ResNet does better

## 6 Conclusions
- **identity shortcut connections** and **identity after-addition activation** 
are essential for making information propagation smooth.