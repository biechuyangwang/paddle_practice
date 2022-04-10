# -*- coding:utf-8 -*-

# 导入需要的包
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class VGG(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(VGG, self).__init__()
        # 有padding的情况下
        # padding = (kernel_size-1)/2
        # out_w = (in_w+2*padding-kernel_size+1)/stride = in_w/stride = 224/4 = 56

        in_channels = [3, 64, 128, 256, 512, 512]
        # 定义第一个block，包含两个卷积
        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        # 定义第二个block，包含两个卷积
        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        # 定义第三个block，包含三个卷积
        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        # 定义第四个block，包含三个卷积
        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        # 定义第五个block，包含三个卷积
        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        # 使用Sequential 将全连接层和relu组成一个线性结构（fc + relu）
        # 当输入为224x224时，经过五个卷积块和池化层后，特征维度变为[512x7x7]
        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(512 * 7 * 7, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential 将全连接层和relu组成一个线性结构（fc + relu）
        self.fc2 = paddle.nn.Sequential(paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())

        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, num_classes)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    # 网络的前向计算过程
    def forward(self, x, label=None):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

# # 打印模型
# from paddle.static import InputSpec
# inputs = InputSpec([None, 3*224*224], 'float32', 'x')
# labels = InputSpec([None, 10], 'float32', 'x')
# model = paddle.Model(VGG(num_classes=10), inputs, labels)
# # 模型可视化
# model.summary((-1,3,224,224))

# ---------------------------------------------------------------------------
#  Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#    Conv2D-1      [[1, 3, 224, 224]]   [1, 64, 224, 224]        1,792
#     ReLU-3          [[1, 4096]]           [1, 4096]              0
#    Conv2D-2     [[1, 64, 224, 224]]   [1, 64, 224, 224]       36,928
#   MaxPool2D-1    [[1, 512, 14, 14]]     [1, 512, 7, 7]           0
#    Conv2D-3     [[1, 64, 112, 112]]   [1, 128, 112, 112]      73,856
#    Conv2D-4     [[1, 128, 112, 112]]  [1, 128, 112, 112]      147,584
#    Conv2D-5      [[1, 128, 56, 56]]    [1, 256, 56, 56]       295,168
#    Conv2D-6      [[1, 256, 56, 56]]    [1, 256, 56, 56]       590,080
#    Conv2D-7      [[1, 256, 56, 56]]    [1, 256, 56, 56]       590,080
#    Conv2D-8      [[1, 256, 28, 28]]    [1, 512, 28, 28]      1,180,160
#    Conv2D-9      [[1, 512, 28, 28]]    [1, 512, 28, 28]      2,359,808
#    Conv2D-10     [[1, 512, 28, 28]]    [1, 512, 28, 28]      2,359,808
#    Conv2D-11     [[1, 512, 14, 14]]    [1, 512, 14, 14]      2,359,808
#    Conv2D-12     [[1, 512, 14, 14]]    [1, 512, 14, 14]      2,359,808
#    Conv2D-13     [[1, 512, 14, 14]]    [1, 512, 14, 14]      2,359,808
#    Linear-1         [[1, 25088]]          [1, 4096]         102,764,544
#     ReLU-1          [[1, 4096]]           [1, 4096]              0
#    Dropout-1        [[1, 4096]]           [1, 4096]              0
#    Linear-2         [[1, 4096]]           [1, 4096]         16,781,312
#     ReLU-2          [[1, 4096]]           [1, 4096]              0
#    Dropout-2        [[1, 4096]]           [1, 4096]              0
#    Linear-3         [[1, 4096]]            [1, 10]            40,970
# ===========================================================================
# Total params: 134,301,514
# Trainable params: 134,301,514
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 103.77
# Params size (MB): 512.32
# Estimated Total Size (MB): 616.66
# ---------------------------------------------------------------------------