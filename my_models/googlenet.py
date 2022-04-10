# -*- coding:utf-8 -*-

# GoogLeNet模型代码
import numpy as np
import paddle
from paddle.nn import Conv2D, MaxPool2D, AdaptiveAvgPool2D, Linear
## 组网
import paddle.nn.functional as F

# 定义Inception块
class Inception(paddle.nn.Layer):
    def __init__(self, c0, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，
        
        c1,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2,图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3,图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list, 
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(in_channels=c0,out_channels=c1, kernel_size=1, stride=1)
        self.p2_1 = Conv2D(in_channels=c0,out_channels=c2[0], kernel_size=1, stride=1)
        self.p2_2 = Conv2D(in_channels=c2[0],out_channels=c2[1], kernel_size=3, padding=1, stride=1)
        self.p3_1 = Conv2D(in_channels=c0,out_channels=c3[0], kernel_size=1, stride=1)
        self.p3_2 = Conv2D(in_channels=c3[0],out_channels=c3[1], kernel_size=5, padding=2, stride=1)
        self.p4_1 = MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2D(in_channels=c0,out_channels=c4, kernel_size=1, stride=1)
        
        # # 新加一层batchnorm稳定收敛
        # self.batchnorm = paddle.nn.BatchNorm2D(c1+c2[1]+c3[1]+c4)

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = F.relu(self.p1_1(x))
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 支路4包含 最大池化和1x1卷积
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return paddle.concat([p1, p2, p3, p4], axis=1)
        # return self.batchnorm()
    
class GoogLeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(GoogLeNet, self).__init__()
        # GoogLeNet包含五个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含1个卷积层
        self.conv1 = Conv2D(in_channels=3,out_channels=64, kernel_size=7, padding=3, stride=1)
        # 3x3最大池化
        self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第二个模块包含2个卷积层
        self.conv2_1 = Conv2D(in_channels=64,out_channels=64, kernel_size=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=64,out_channels=192, kernel_size=3, padding=1, stride=1)
        # 3x3最大池化
        self.pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        # 3x3最大池化
        self.pool3 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        # 3x3最大池化
        self.pool4 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        # 全局池化，用的是global_pooling，不需要设置pool_stride
        self.pool5 = AdaptiveAvgPool2D(output_size=1)
        self.fc = Linear(in_features=1024, out_features=num_classes)

    def forward(self, x, label=None):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.pool3(self.block3_2(self.block3_1(x)))
        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.pool4(self.block4_5(self.block4_4(x)))
        x = self.pool5(self.block5_2(self.block5_1(x)))
        # x = paddle.reshape(x, [x.shape[0], -1])
        x = paddle.flatten(x, 1, -1)
        x = self.fc(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

# # 打印模型
# from paddle.static import InputSpec
# inputs = InputSpec([None, 3*224*224], 'float32', 'x')
# labels = InputSpec([None, 10], 'float32', 'x')
# model = paddle.Model(GoogLeNet(num_classes=10), inputs, labels)
# # 模型可视化
# model.summary((-1,3,224,224))

# -------------------------------------------------------------------------------
#    Layer (type)         Input Shape          Output Shape         Param #
# ===============================================================================
#      Conv2D-1        [[1, 3, 224, 224]]   [1, 64, 224, 224]        9,472
#     MaxPool2D-1     [[1, 64, 224, 224]]   [1, 64, 112, 112]          0
#      Conv2D-2       [[1, 64, 112, 112]]   [1, 64, 112, 112]        4,160
#      Conv2D-3       [[1, 64, 112, 112]]   [1, 192, 112, 112]      110,784
#     MaxPool2D-2     [[1, 192, 112, 112]]   [1, 192, 56, 56]          0
#      Conv2D-4        [[1, 192, 56, 56]]    [1, 64, 56, 56]        12,352
#      Conv2D-5        [[1, 192, 56, 56]]    [1, 96, 56, 56]        18,528
#      Conv2D-6        [[1, 96, 56, 56]]     [1, 128, 56, 56]       110,720
#      Conv2D-7        [[1, 192, 56, 56]]    [1, 16, 56, 56]         3,088
#      Conv2D-8        [[1, 16, 56, 56]]     [1, 32, 56, 56]        12,832
#     MaxPool2D-3      [[1, 192, 56, 56]]    [1, 192, 56, 56]          0
#      Conv2D-9        [[1, 192, 56, 56]]    [1, 32, 56, 56]         6,176
#     Inception-1      [[1, 192, 56, 56]]    [1, 256, 56, 56]          0
#      Conv2D-10       [[1, 256, 56, 56]]    [1, 128, 56, 56]       32,896
#      Conv2D-11       [[1, 256, 56, 56]]    [1, 128, 56, 56]       32,896
#      Conv2D-12       [[1, 128, 56, 56]]    [1, 192, 56, 56]       221,376
#      Conv2D-13       [[1, 256, 56, 56]]    [1, 32, 56, 56]         8,224
#      Conv2D-14       [[1, 32, 56, 56]]     [1, 96, 56, 56]        76,896
#     MaxPool2D-4      [[1, 256, 56, 56]]    [1, 256, 56, 56]          0
#      Conv2D-15       [[1, 256, 56, 56]]    [1, 64, 56, 56]        16,448
#     Inception-2      [[1, 256, 56, 56]]    [1, 480, 56, 56]          0
#     MaxPool2D-5      [[1, 480, 56, 56]]    [1, 480, 28, 28]          0
#      Conv2D-16       [[1, 480, 28, 28]]    [1, 192, 28, 28]       92,352
#      Conv2D-17       [[1, 480, 28, 28]]    [1, 96, 28, 28]        46,176
#      Conv2D-18       [[1, 96, 28, 28]]     [1, 208, 28, 28]       179,920
#      Conv2D-19       [[1, 480, 28, 28]]    [1, 16, 28, 28]         7,696
#      Conv2D-20       [[1, 16, 28, 28]]     [1, 48, 28, 28]        19,248
#     MaxPool2D-6      [[1, 480, 28, 28]]    [1, 480, 28, 28]          0
#      Conv2D-21       [[1, 480, 28, 28]]    [1, 64, 28, 28]        30,784
#     Inception-3      [[1, 480, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-22       [[1, 512, 28, 28]]    [1, 160, 28, 28]       82,080
#      Conv2D-23       [[1, 512, 28, 28]]    [1, 112, 28, 28]       57,456
#      Conv2D-24       [[1, 112, 28, 28]]    [1, 224, 28, 28]       226,016
#      Conv2D-25       [[1, 512, 28, 28]]    [1, 24, 28, 28]        12,312
#      Conv2D-26       [[1, 24, 28, 28]]     [1, 64, 28, 28]        38,464
#     MaxPool2D-7      [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-27       [[1, 512, 28, 28]]    [1, 64, 28, 28]        32,832
#     Inception-4      [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-28       [[1, 512, 28, 28]]    [1, 128, 28, 28]       65,664
#      Conv2D-29       [[1, 512, 28, 28]]    [1, 128, 28, 28]       65,664
#      Conv2D-30       [[1, 128, 28, 28]]    [1, 256, 28, 28]       295,168
#      Conv2D-31       [[1, 512, 28, 28]]    [1, 24, 28, 28]        12,312
#      Conv2D-32       [[1, 24, 28, 28]]     [1, 64, 28, 28]        38,464
#     MaxPool2D-8      [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-33       [[1, 512, 28, 28]]    [1, 64, 28, 28]        32,832
#     Inception-5      [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-34       [[1, 512, 28, 28]]    [1, 112, 28, 28]       57,456
#      Conv2D-35       [[1, 512, 28, 28]]    [1, 144, 28, 28]       73,872
#      Conv2D-36       [[1, 144, 28, 28]]    [1, 288, 28, 28]       373,536
#      Conv2D-37       [[1, 512, 28, 28]]    [1, 32, 28, 28]        16,416
#      Conv2D-38       [[1, 32, 28, 28]]     [1, 64, 28, 28]        51,264
#     MaxPool2D-9      [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-39       [[1, 512, 28, 28]]    [1, 64, 28, 28]        32,832
#     Inception-6      [[1, 512, 28, 28]]    [1, 528, 28, 28]          0
#      Conv2D-40       [[1, 528, 28, 28]]    [1, 256, 28, 28]       135,424
#      Conv2D-41       [[1, 528, 28, 28]]    [1, 160, 28, 28]       84,640
#      Conv2D-42       [[1, 160, 28, 28]]    [1, 320, 28, 28]       461,120
#      Conv2D-43       [[1, 528, 28, 28]]    [1, 32, 28, 28]        16,928
#      Conv2D-44       [[1, 32, 28, 28]]     [1, 128, 28, 28]       102,528
#    MaxPool2D-10      [[1, 528, 28, 28]]    [1, 528, 28, 28]          0
#      Conv2D-45       [[1, 528, 28, 28]]    [1, 128, 28, 28]       67,712
#     Inception-7      [[1, 528, 28, 28]]    [1, 832, 28, 28]          0
#    MaxPool2D-11      [[1, 832, 28, 28]]    [1, 832, 14, 14]          0
#      Conv2D-46       [[1, 832, 14, 14]]    [1, 256, 14, 14]       213,248
#      Conv2D-47       [[1, 832, 14, 14]]    [1, 160, 14, 14]       133,280
#      Conv2D-48       [[1, 160, 14, 14]]    [1, 320, 14, 14]       461,120
#      Conv2D-49       [[1, 832, 14, 14]]    [1, 32, 14, 14]        26,656
#      Conv2D-50       [[1, 32, 14, 14]]     [1, 128, 14, 14]       102,528
#    MaxPool2D-12      [[1, 832, 14, 14]]    [1, 832, 14, 14]          0
#      Conv2D-51       [[1, 832, 14, 14]]    [1, 128, 14, 14]       106,624
#     Inception-8      [[1, 832, 14, 14]]    [1, 832, 14, 14]          0
#      Conv2D-52       [[1, 832, 14, 14]]    [1, 384, 14, 14]       319,872
#      Conv2D-53       [[1, 832, 14, 14]]    [1, 192, 14, 14]       159,936
#      Conv2D-54       [[1, 192, 14, 14]]    [1, 384, 14, 14]       663,936
#      Conv2D-55       [[1, 832, 14, 14]]    [1, 48, 14, 14]        39,984
#      Conv2D-56       [[1, 48, 14, 14]]     [1, 128, 14, 14]       153,728
#    MaxPool2D-13      [[1, 832, 14, 14]]    [1, 832, 14, 14]          0
#      Conv2D-57       [[1, 832, 14, 14]]    [1, 128, 14, 14]       106,624
#     Inception-9      [[1, 832, 14, 14]]   [1, 1024, 14, 14]          0
# AdaptiveAvgPool2D-1 [[1, 1024, 14, 14]]    [1, 1024, 1, 1]           0
#      Linear-1           [[1, 1024]]            [1, 10]            10,250
# ===============================================================================
# Total params: 5,983,802
# Trainable params: 5,983,802
# Non-trainable params: 0
# -------------------------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 179.43
# Params size (MB): 22.83
# Estimated Total Size (MB): 202.83
# -------------------------------------------------------------------------------