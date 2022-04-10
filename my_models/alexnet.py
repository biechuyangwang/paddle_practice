# -*- coding:utf-8 -*-

# 导入需要的包
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class AlexNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        # 有padding的情况下
        # padding = (kernel_size-1)/2
        # out_w = (in_w+2*padding-kernel_size+1)/stride = in_w/stride = 224/4 = 56

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5) # 3*224*224 => 96*56*56
        self.relu1 = paddle.nn.ReLU()
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # 96*56*56 => 96*28*28

        self.conv2 = Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2) # 96*28*28 => 256*28*28
        self.relu2 = paddle.nn.ReLU()
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # 256*28*28 => 256*14*14

        # 创建第3/4/5个卷积层
        self.conv3 = Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1) # 256*14*14 => 384*14*14
        self.relu3 = paddle.nn.ReLU()
        self.conv4 = Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) # 384*14*14 => 384*14*14
        self.relu4 = paddle.nn.ReLU()
        self.conv5 = Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1) # 384*14*14 => 256*14*14
        self.relu5 = paddle.nn.ReLU()
        self.max_pool5 = MaxPool2D(kernel_size=2, stride=2) # 256*14*14 => 256*7*7
        
        # 创建全连接层，第一个全连接层的输出神经元个数为4096
        self.fc1 = Linear(in_features=256*7*7, out_features=4096) # in_features需要计算，所以in_features=120*50*50
        self.relu6 = paddle.nn.ReLU()
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')

        # 创建全连接层，第二个全连接层的输出神经元个数为4096
        self.fc2 = Linear(in_features=4096, out_features=4096)
        self.relu7 = paddle.nn.ReLU()
        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')

        # 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc3 = Linear(in_features=4096, out_features=num_classes)

        # 公用部分，在设计时，最好向上面一样一层层设计，不要使用公共的，但是最后设计完了再精简成下面的形式
        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    # 网络的前向计算过程
    def forward(self, x, label=None):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x) 

        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.max_pool5(x)

        x = paddle.flatten(x, 1, -1) # (x,start_axis-1,stop_axis=-1)
        x = self.dropout1(self.relu6(self.fc1(x)))
        x = self.dropout2(self.relu7(self.fc2(x)))
        x = self.fc3(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

# 打印模型
from paddle.static import InputSpec
inputs = InputSpec([None, 3*224*224], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(AlexNet(num_classes=10), inputs, labels)
# 模型可视化
model.summary((-1,3,224,224))

# ---------------------------------------------------------------------------
#  Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#    Conv2D-1      [[1, 3, 224, 224]]    [1, 96, 56, 56]        34,944
#     ReLU-1       [[1, 96, 56, 56]]     [1, 96, 56, 56]           0
#   MaxPool2D-1    [[1, 96, 56, 56]]     [1, 96, 28, 28]           0
#    Conv2D-2      [[1, 96, 28, 28]]     [1, 256, 28, 28]       614,656
#     ReLU-2       [[1, 256, 28, 28]]    [1, 256, 28, 28]          0
#   MaxPool2D-2    [[1, 256, 28, 28]]    [1, 256, 14, 14]          0
#    Conv2D-3      [[1, 256, 14, 14]]    [1, 384, 14, 14]       885,120
#     ReLU-3       [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#    Conv2D-4      [[1, 384, 14, 14]]    [1, 384, 14, 14]      1,327,488
#     ReLU-4       [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#    Conv2D-5      [[1, 384, 14, 14]]    [1, 256, 14, 14]       884,992
#     ReLU-5       [[1, 256, 14, 14]]    [1, 256, 14, 14]          0
#   MaxPool2D-3    [[1, 256, 14, 14]]     [1, 256, 7, 7]           0
#    Linear-1         [[1, 12544]]          [1, 4096]         51,384,320
#     ReLU-6          [[1, 4096]]           [1, 4096]              0
#    Dropout-1        [[1, 4096]]           [1, 4096]              0
#    Linear-2         [[1, 4096]]           [1, 4096]         16,781,312
#     ReLU-7          [[1, 4096]]           [1, 4096]              0
#    Dropout-2        [[1, 4096]]           [1, 4096]              0
#    Linear-3         [[1, 4096]]            [1, 10]            40,970
# ===========================================================================
# Total params: 71,953,802
# Trainable params: 71,953,802
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 11.96
# Params size (MB): 274.48
# Estimated Total Size (MB): 287.02
# ---------------------------------------------------------------------------