# -*- coding:utf-8 -*-

# 导入需要的包
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(in_channels=3, out_channels=6, kernel_size=5) # 3*224*224 => 6*220*220
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # 6*220*220 => 6*110*110
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5) # 6*110*110 => 16*106*106
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # 16*106*106 => 16*53*53
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4) # 16*53*53 => 120*50*50
        # 创建全连接层，第一个全连接层的输出神经元个数为64
        self.fc1 = Linear(in_features=120*50*50, out_features=64) # in_features需要计算, ((224-4)/2-4)/2-3=50 所以in_features=120*50*50
        # 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    # 网络的前向计算过程
    def forward(self, x, label=None):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)

        x = paddle.flatten(x, 1, -1) # (x,start_axis-1,stop_axis=-1) # 与下面的操作等价，在加速的使用原生的reshape会好一点
        # x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

# # 打印模型
# from paddle.static import InputSpec
# inputs = InputSpec([None, 3*224*224], 'float32', 'x')
# labels = InputSpec([None, 10], 'float32', 'x')
# model = paddle.Model(LeNet(num_classes=10), inputs, labels)
# # 模型可视化
# model.summary((-1,3,224,224))

# ---------------------------------------------------------------------------
#  Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#    Conv2D-1      [[1, 3, 224, 224]]    [1, 6, 220, 220]         456
#   MaxPool2D-1    [[1, 6, 220, 220]]    [1, 6, 110, 110]          0
#    Conv2D-2      [[1, 6, 110, 110]]   [1, 16, 106, 106]        2,416
#   MaxPool2D-2   [[1, 16, 106, 106]]    [1, 16, 53, 53]           0
#    Conv2D-3      [[1, 16, 53, 53]]     [1, 120, 50, 50]       30,840
#    Linear-1        [[1, 300000]]           [1, 64]          19,200,064
#    Linear-2          [[1, 64]]             [1, 10]              650
# ===========================================================================
# Total params: 19,234,426
# Trainable params: 19,234,426
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.77
# Params size (MB): 73.37
# Estimated Total Size (MB): 80.72
# ---------------------------------------------------------------------------