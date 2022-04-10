# 数据预处理和训练中的tricks
## 0 经典tricks
### 0.1 L1/L2正则 (损失函数中的权重衰减)
### 0.2 Batch Normalization (mini-batch的卷积层后面)
### 0.3 Early Stopping (避免浪费计算资源)
### 0.4 Random Cropping
### 0.5 Mirroring
### 0.6 Rotation
### 0.7 Color shifting
### 0.8 Xavier init

## 1 Warmup
### 1.1 背景
学习率是神经网络训练中最重要的超参数之一，针对学习率的技巧有很多。Warm up是在ResNet论文中提到的一种学习率预热的方法。由于刚开始训练时模型的权重(weights)是随机初始化的，此时选择一个较大的学习率，可能会带来模型的不稳定。

### 1.2 解决方法
学习率预热就是在刚开始训练的时候先使用一个较小的学习率，训练一些epoches或iterations，等模型稳定时再修改为预先设置的学习率进行训练。

- constant warmup

    先用0.01的学习率训练直到训练误差低于80%(大概训练了400个iterations)，然后使用0.1的学习率进行训练

- gradual warmup

    因为从一个很小的学习率一下变为比较大的学习率可能会导致训练误差突然增大。
    
    gradual warmup来解决这个问题，即从最开始的小学习率开始，每个iteration增大一点，直到最初设置的比较大的学习率。

## 2 Linear scaling learning rate
### 2.1 背景
在凸优化问题中，随着批量的增加，收敛速度会降低，神经网络也有类似的实证结果。随着batch size的增大，处理相同数据量的速度会越来越快，但是达到相同精度所需要的epoch数量越来越多。也就是说，使用相同的epoch时，大batch size训练的模型与小batch size训练的模型相比，验证准确率会减小。

### 2.2 解决办法
上面提到的gradual warmup是解决此问题的方法之一。

另外，linear scaling learning rate也是一种有效的方法。在mini-batch SGD训练时，梯度下降的值是随机的，因为每一个batch的数据是随机选择的。增大batch size不会改变梯度的期望，但是会降低它的方差。也就是说，大batch size会降低梯度中的噪声，所以我们可以增大学习率来加快收敛。

具体做法很简单，比如ResNet原论文中，batch size为256时选择的学习率是0.1，当我们把batch size变为一个较大的数b时，学习率应该变为 0.1 × b/256。

## 3 Label-smoothing
### 3.1 背景
在分类问题中，我们的最后一层一般是全连接层，然后对应标签的one-hot编码，即把对应类别的值编码为1，其他为0。这种编码方式和通过降低交叉熵损失来调整参数的方式结合起来，会有一些问题。这种方式会鼓励模型对不同类别的输出分数差异非常大，或者说，模型过分相信它的判断。但是，对于一个由多人标注的数据集，不同人标注的准则可能不同，每个人的标注也可能会有一些错误。模型对标签的过分相信会导致过拟合。

### 3.2 解决办法
标签平滑(Label-smoothing regularization,LSR)是应对该问题的有效方法之一，它的具体思想是降低我们对于标签的信任，例如我们可以将损失的目标值从1稍微降到0.9，或者将从0稍微升到0.1。
$$
\begin{equation}
    q_i=
    \left\{
        \begin{aligned}
            & 1-\epsilon & & {if\ i=j} \\
            & \epsilon/(K-1) & & {otherwise}
        \end{aligned}
    \right.
\end{equation}
$$
其中，$\epsilon$是一个小的常数，$K$是类别的数目，y是图片的真正的标签，i代表第i个类别，q_i是图片为第i类的概率。

总的来说，LSR是一种通过在标签y中加入噪声，实现对模型约束，降低模型过拟合程度的一种正则化方法

## 4 Random image cropping and patching
### 4.1 背景
### 4.2 解决办法
Random image cropping and patching (RICAP)方法随机裁剪四个图片的中部分，然后把它们拼接为一个图片，同时混合这四个图片的标签。超参数$\beta$控制裁剪的尺寸（裁剪的宽高来自beta分布）

## 5 Knowledge Distallation(知识蒸馏)
### 5.1 背景
提高几乎所有机器学习算法性能的一种非常简单的方法是在相同的数据上训练许多不同的模型，然后对它们的预测进行平均。但是使用所有的模型集成进行预测是比较麻烦的，并且可能计算量太大而无法部署到大量用户。
### 5.2 解决办法
在知识蒸馏方法中，我们使用一个教师模型来帮助当前的模型（学生模型）训练。教师模型是一个较高准确率的预训练模型，因此学生模型可以在保持模型复杂度不变的情况下提升准确率。比如用resnet-152来训练mobile-net来提高mobile-net精度。

## 6 Cutout
### 6.1 背景
### 6.2 解决办法
Cutout是一种新的正则化方法。原理是在训练时随机把图片的一部分减掉，这样能提高模型的鲁棒性。它的来源是计算机视觉任务中经常遇到的物体遮挡问题。通过cutout生成一些类似被遮挡的物体，不仅可以让模型在遇到遮挡问题时表现更好，还能让模型在做决定时更多地考虑环境(context)。

## 7 Random erasing
### 7.1 背景
### 7.2 解决办法
Random erasing其实和cutout非常类似，也是一种模拟物体遮挡情况的数据增强方法。区别在于，cutout是把图片中随机抽中的矩形区域的像素值置为0，相当于裁剪掉，random erasing是用随机数或者数据集中像素的平均值替换原来的像素值。而且，cutout每次裁剪掉的区域大小是固定的，Random erasing替换掉的区域大小是随机的。

## 8 Cosine learning rate decay
### 8.1 背景
### 8.2 解决办法
在warmup之后的训练过程中，学习率不断衰减是一个提高精度的好方法。其中有step decay和cosine decay等，前者是随着epoch增大学习率不断减去一个小的数，后者是让学习率随着训练过程曲线下降。

对于cosine decay，假设总共有$T$个batch（不考虑warmup阶段），在第$t$个batch时，学习率$\eta_t$为：
$$
\begin{equation}
    \eta_t=\frac{1}{2}
    \left(
        1+\cos\left( \frac{t\pi}{T} \right)
    \right)
    \eta
\end{equation}
$$

## 9 Mixup training
### 9.1 背景
### 9.2 解决办法
Mixup是一种新的数据增强的方法。Mixup training，就是每次取出2张图片，然后将它们线性组合，得到新的图片，以此来作为新的训练样本，进行网络的训练。$\lambda$是超参数，控住样本和标签。
$$
\begin{equation} 
    \begin{split}
        \hat{x}=\lambda x_i+(1-\lambda)x_j, \\
        \hat{y}=\lambda y_i+(1-\lambda)y_j
    \end{split}
\end{equation}
$$

## 10 AdaBound
### 10.1 背景
### 10.2 解决办法
AdaBound会让你的训练过程像adam一样快，并且像SGD一样好。

## 11 AutoAugment
### 11.1 背景
数据增强在图像分类问题上有很重要的作用，但是增强的方法有很多，并非一股脑地用上所有的方法就是最好的。那么，如何选择最佳的数据增强方法呢？
### 11.2 解决办法
AutoAugment就是一种搜索适合当前问题的数据增强方法的方法。该方法创建一个数据增强策略的搜索空间，利用搜索算法选取适合特定数据集的数据增强策略。此外，从一个数据集中学到的策略能够很好地迁移到其它相似的数据集上。