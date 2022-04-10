# -*- coding:utf-8 -*-

# ResNet模型代码
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
       
        """
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)
        
        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y

# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, num_classes=1):
        """
        
        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]
        
        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1)

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, 
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接的输入维度是2048
        self.out = nn.Linear(in_features=2048, out_features=num_classes,
                      weight_attr=paddle.ParamAttr(
                          initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs, label=None):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        # y = paddle.reshape(y, [y.shape[0], -1])
        y = paddle.flatten(y, 1, -1)
        y = self.out(y)
        if label is not None:
            acc = paddle.metric.accuracy(input=y, label=label)
            return y, acc
        else:
            return y

# 打印模型
from paddle.vision.models import resnet50, vgg16, LeNet, mobilenet_v2
from paddle.static import InputSpec
inputs = InputSpec([None, 3*224*224], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(ResNet(num_classes=10), inputs, labels)
# model = paddle.Model(mobilenet_v2(num_classes=10),inputs,labels)
# 模型可视化
model.summary((-1,3,224,224))

# ResNet50
# -------------------------------------------------------------------------------
#    Layer (type)         Input Shape          Output Shape         Param #
# ===============================================================================
#      Conv2D-1        [[1, 3, 224, 224]]   [1, 64, 112, 112]        9,408
#    BatchNorm2D-1    [[1, 64, 112, 112]]   [1, 64, 112, 112]         256
#    ConvBNLayer-1     [[1, 3, 224, 224]]   [1, 64, 112, 112]          0
#     MaxPool2D-1     [[1, 64, 112, 112]]    [1, 64, 56, 56]           0
#      Conv2D-2        [[1, 64, 56, 56]]     [1, 64, 56, 56]         4,096
#    BatchNorm2D-2     [[1, 64, 56, 56]]     [1, 64, 56, 56]          256
#    ConvBNLayer-2     [[1, 64, 56, 56]]     [1, 64, 56, 56]           0
#      Conv2D-3        [[1, 64, 56, 56]]     [1, 64, 56, 56]        36,864
#    BatchNorm2D-3     [[1, 64, 56, 56]]     [1, 64, 56, 56]          256
#    ConvBNLayer-3     [[1, 64, 56, 56]]     [1, 64, 56, 56]           0
#      Conv2D-4        [[1, 64, 56, 56]]     [1, 256, 56, 56]       16,384
#    BatchNorm2D-4     [[1, 256, 56, 56]]    [1, 256, 56, 56]        1,024
#    ConvBNLayer-4     [[1, 64, 56, 56]]     [1, 256, 56, 56]          0
#      Conv2D-5        [[1, 64, 56, 56]]     [1, 256, 56, 56]       16,384
#    BatchNorm2D-5     [[1, 256, 56, 56]]    [1, 256, 56, 56]        1,024
#    ConvBNLayer-5     [[1, 64, 56, 56]]     [1, 256, 56, 56]          0
#  BottleneckBlock-1   [[1, 64, 56, 56]]     [1, 256, 56, 56]          0
#      Conv2D-6        [[1, 256, 56, 56]]    [1, 64, 56, 56]        16,384
#    BatchNorm2D-6     [[1, 64, 56, 56]]     [1, 64, 56, 56]          256
#    ConvBNLayer-6     [[1, 256, 56, 56]]    [1, 64, 56, 56]           0
#      Conv2D-7        [[1, 64, 56, 56]]     [1, 64, 56, 56]        36,864
#    BatchNorm2D-7     [[1, 64, 56, 56]]     [1, 64, 56, 56]          256
#    ConvBNLayer-7     [[1, 64, 56, 56]]     [1, 64, 56, 56]           0
#      Conv2D-8        [[1, 64, 56, 56]]     [1, 256, 56, 56]       16,384
#    BatchNorm2D-8     [[1, 256, 56, 56]]    [1, 256, 56, 56]        1,024
#    ConvBNLayer-8     [[1, 64, 56, 56]]     [1, 256, 56, 56]          0
#  BottleneckBlock-2   [[1, 256, 56, 56]]    [1, 256, 56, 56]          0
#      Conv2D-9        [[1, 256, 56, 56]]    [1, 64, 56, 56]        16,384
#    BatchNorm2D-9     [[1, 64, 56, 56]]     [1, 64, 56, 56]          256
#    ConvBNLayer-9     [[1, 256, 56, 56]]    [1, 64, 56, 56]           0
#      Conv2D-10       [[1, 64, 56, 56]]     [1, 64, 56, 56]        36,864
#   BatchNorm2D-10     [[1, 64, 56, 56]]     [1, 64, 56, 56]          256
#   ConvBNLayer-10     [[1, 64, 56, 56]]     [1, 64, 56, 56]           0
#      Conv2D-11       [[1, 64, 56, 56]]     [1, 256, 56, 56]       16,384
#   BatchNorm2D-11     [[1, 256, 56, 56]]    [1, 256, 56, 56]        1,024
#   ConvBNLayer-11     [[1, 64, 56, 56]]     [1, 256, 56, 56]          0
#  BottleneckBlock-3   [[1, 256, 56, 56]]    [1, 256, 56, 56]          0
#      Conv2D-12       [[1, 256, 56, 56]]    [1, 128, 56, 56]       32,768
#   BatchNorm2D-12     [[1, 128, 56, 56]]    [1, 128, 56, 56]         512
#   ConvBNLayer-12     [[1, 256, 56, 56]]    [1, 128, 56, 56]          0
#      Conv2D-13       [[1, 128, 56, 56]]    [1, 128, 28, 28]       147,456
#   BatchNorm2D-13     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-13     [[1, 128, 56, 56]]    [1, 128, 28, 28]          0
#      Conv2D-14       [[1, 128, 28, 28]]    [1, 512, 28, 28]       65,536
#   BatchNorm2D-14     [[1, 512, 28, 28]]    [1, 512, 28, 28]        2,048
#   ConvBNLayer-14     [[1, 128, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-15       [[1, 256, 56, 56]]    [1, 512, 28, 28]       131,072
#   BatchNorm2D-15     [[1, 512, 28, 28]]    [1, 512, 28, 28]        2,048
#   ConvBNLayer-15     [[1, 256, 56, 56]]    [1, 512, 28, 28]          0
#  BottleneckBlock-4   [[1, 256, 56, 56]]    [1, 512, 28, 28]          0
#      Conv2D-16       [[1, 512, 28, 28]]    [1, 128, 28, 28]       65,536
#   BatchNorm2D-16     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-16     [[1, 512, 28, 28]]    [1, 128, 28, 28]          0
#      Conv2D-17       [[1, 128, 28, 28]]    [1, 128, 28, 28]       147,456
#   BatchNorm2D-17     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-17     [[1, 128, 28, 28]]    [1, 128, 28, 28]          0
#      Conv2D-18       [[1, 128, 28, 28]]    [1, 512, 28, 28]       65,536
#   BatchNorm2D-18     [[1, 512, 28, 28]]    [1, 512, 28, 28]        2,048
#   ConvBNLayer-18     [[1, 128, 28, 28]]    [1, 512, 28, 28]          0
#  BottleneckBlock-5   [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-19       [[1, 512, 28, 28]]    [1, 128, 28, 28]       65,536
#   BatchNorm2D-19     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-19     [[1, 512, 28, 28]]    [1, 128, 28, 28]          0
#      Conv2D-20       [[1, 128, 28, 28]]    [1, 128, 28, 28]       147,456
#   BatchNorm2D-20     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-20     [[1, 128, 28, 28]]    [1, 128, 28, 28]          0
#      Conv2D-21       [[1, 128, 28, 28]]    [1, 512, 28, 28]       65,536
#   BatchNorm2D-21     [[1, 512, 28, 28]]    [1, 512, 28, 28]        2,048
#   ConvBNLayer-21     [[1, 128, 28, 28]]    [1, 512, 28, 28]          0
#  BottleneckBlock-6   [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-22       [[1, 512, 28, 28]]    [1, 128, 28, 28]       65,536
#   BatchNorm2D-22     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-22     [[1, 512, 28, 28]]    [1, 128, 28, 28]          0
#      Conv2D-23       [[1, 128, 28, 28]]    [1, 128, 28, 28]       147,456
#   BatchNorm2D-23     [[1, 128, 28, 28]]    [1, 128, 28, 28]         512
#   ConvBNLayer-23     [[1, 128, 28, 28]]    [1, 128, 28, 28]          0
#      Conv2D-24       [[1, 128, 28, 28]]    [1, 512, 28, 28]       65,536
#   BatchNorm2D-24     [[1, 512, 28, 28]]    [1, 512, 28, 28]        2,048
#   ConvBNLayer-24     [[1, 128, 28, 28]]    [1, 512, 28, 28]          0
#  BottleneckBlock-7   [[1, 512, 28, 28]]    [1, 512, 28, 28]          0
#      Conv2D-25       [[1, 512, 28, 28]]    [1, 256, 28, 28]       131,072
#   BatchNorm2D-25     [[1, 256, 28, 28]]    [1, 256, 28, 28]        1,024
#   ConvBNLayer-25     [[1, 512, 28, 28]]    [1, 256, 28, 28]          0
#      Conv2D-26       [[1, 256, 28, 28]]    [1, 256, 14, 14]       589,824
#   BatchNorm2D-26     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-26     [[1, 256, 28, 28]]    [1, 256, 14, 14]          0
#      Conv2D-27       [[1, 256, 14, 14]]   [1, 1024, 14, 14]       262,144
#   BatchNorm2D-27    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-27     [[1, 256, 14, 14]]   [1, 1024, 14, 14]          0
#      Conv2D-28       [[1, 512, 28, 28]]   [1, 1024, 14, 14]       524,288
#   BatchNorm2D-28    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-28     [[1, 512, 28, 28]]   [1, 1024, 14, 14]          0
#  BottleneckBlock-8   [[1, 512, 28, 28]]   [1, 1024, 14, 14]          0
#      Conv2D-29      [[1, 1024, 14, 14]]    [1, 256, 14, 14]       262,144
#   BatchNorm2D-29     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-29    [[1, 1024, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-30       [[1, 256, 14, 14]]    [1, 256, 14, 14]       589,824
#   BatchNorm2D-30     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-30     [[1, 256, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-31       [[1, 256, 14, 14]]   [1, 1024, 14, 14]       262,144
#   BatchNorm2D-31    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-31     [[1, 256, 14, 14]]   [1, 1024, 14, 14]          0
#  BottleneckBlock-9  [[1, 1024, 14, 14]]   [1, 1024, 14, 14]          0
#      Conv2D-32      [[1, 1024, 14, 14]]    [1, 256, 14, 14]       262,144
#   BatchNorm2D-32     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-32    [[1, 1024, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-33       [[1, 256, 14, 14]]    [1, 256, 14, 14]       589,824
#   BatchNorm2D-33     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-33     [[1, 256, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-34       [[1, 256, 14, 14]]   [1, 1024, 14, 14]       262,144
#   BatchNorm2D-34    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-34     [[1, 256, 14, 14]]   [1, 1024, 14, 14]          0
# BottleneckBlock-10  [[1, 1024, 14, 14]]   [1, 1024, 14, 14]          0
#      Conv2D-35      [[1, 1024, 14, 14]]    [1, 256, 14, 14]       262,144
#   BatchNorm2D-35     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-35    [[1, 1024, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-36       [[1, 256, 14, 14]]    [1, 256, 14, 14]       589,824
#   BatchNorm2D-36     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-36     [[1, 256, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-37       [[1, 256, 14, 14]]   [1, 1024, 14, 14]       262,144
#   BatchNorm2D-37    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-37     [[1, 256, 14, 14]]   [1, 1024, 14, 14]          0
# BottleneckBlock-11  [[1, 1024, 14, 14]]   [1, 1024, 14, 14]          0
#      Conv2D-38      [[1, 1024, 14, 14]]    [1, 256, 14, 14]       262,144
#   BatchNorm2D-38     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-38    [[1, 1024, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-39       [[1, 256, 14, 14]]    [1, 256, 14, 14]       589,824
#   BatchNorm2D-39     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-39     [[1, 256, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-40       [[1, 256, 14, 14]]   [1, 1024, 14, 14]       262,144
#   BatchNorm2D-40    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-40     [[1, 256, 14, 14]]   [1, 1024, 14, 14]          0
# BottleneckBlock-12  [[1, 1024, 14, 14]]   [1, 1024, 14, 14]          0
#      Conv2D-41      [[1, 1024, 14, 14]]    [1, 256, 14, 14]       262,144
#   BatchNorm2D-41     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-41    [[1, 1024, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-42       [[1, 256, 14, 14]]    [1, 256, 14, 14]       589,824
#   BatchNorm2D-42     [[1, 256, 14, 14]]    [1, 256, 14, 14]        1,024
#   ConvBNLayer-42     [[1, 256, 14, 14]]    [1, 256, 14, 14]          0
#      Conv2D-43       [[1, 256, 14, 14]]   [1, 1024, 14, 14]       262,144
#   BatchNorm2D-43    [[1, 1024, 14, 14]]   [1, 1024, 14, 14]        4,096
#   ConvBNLayer-43     [[1, 256, 14, 14]]   [1, 1024, 14, 14]          0
# BottleneckBlock-13  [[1, 1024, 14, 14]]   [1, 1024, 14, 14]          0
#      Conv2D-44      [[1, 1024, 14, 14]]    [1, 512, 14, 14]       524,288
#   BatchNorm2D-44     [[1, 512, 14, 14]]    [1, 512, 14, 14]        2,048
#   ConvBNLayer-44    [[1, 1024, 14, 14]]    [1, 512, 14, 14]          0
#      Conv2D-45       [[1, 512, 14, 14]]     [1, 512, 7, 7]       2,359,296
#   BatchNorm2D-45      [[1, 512, 7, 7]]      [1, 512, 7, 7]         2,048
#   ConvBNLayer-45     [[1, 512, 14, 14]]     [1, 512, 7, 7]           0
#      Conv2D-46        [[1, 512, 7, 7]]     [1, 2048, 7, 7]       1,048,576
#   BatchNorm2D-46     [[1, 2048, 7, 7]]     [1, 2048, 7, 7]         8,192
#   ConvBNLayer-46      [[1, 512, 7, 7]]     [1, 2048, 7, 7]           0
#      Conv2D-47      [[1, 1024, 14, 14]]    [1, 2048, 7, 7]       2,097,152
#   BatchNorm2D-47     [[1, 2048, 7, 7]]     [1, 2048, 7, 7]         8,192
#   ConvBNLayer-47    [[1, 1024, 14, 14]]    [1, 2048, 7, 7]           0
# BottleneckBlock-14  [[1, 1024, 14, 14]]    [1, 2048, 7, 7]           0
#      Conv2D-48       [[1, 2048, 7, 7]]      [1, 512, 7, 7]       1,048,576
#   BatchNorm2D-48      [[1, 512, 7, 7]]      [1, 512, 7, 7]         2,048
#   ConvBNLayer-48     [[1, 2048, 7, 7]]      [1, 512, 7, 7]           0
#      Conv2D-49        [[1, 512, 7, 7]]      [1, 512, 7, 7]       2,359,296
#   BatchNorm2D-49      [[1, 512, 7, 7]]      [1, 512, 7, 7]         2,048
#   ConvBNLayer-49      [[1, 512, 7, 7]]      [1, 512, 7, 7]           0
#      Conv2D-50        [[1, 512, 7, 7]]     [1, 2048, 7, 7]       1,048,576
#   BatchNorm2D-50     [[1, 2048, 7, 7]]     [1, 2048, 7, 7]         8,192
#   ConvBNLayer-50      [[1, 512, 7, 7]]     [1, 2048, 7, 7]           0
# BottleneckBlock-15   [[1, 2048, 7, 7]]     [1, 2048, 7, 7]           0
#      Conv2D-51       [[1, 2048, 7, 7]]      [1, 512, 7, 7]       1,048,576
#   BatchNorm2D-51      [[1, 512, 7, 7]]      [1, 512, 7, 7]         2,048
#   ConvBNLayer-51     [[1, 2048, 7, 7]]      [1, 512, 7, 7]           0
#      Conv2D-52        [[1, 512, 7, 7]]      [1, 512, 7, 7]       2,359,296
#   BatchNorm2D-52      [[1, 512, 7, 7]]      [1, 512, 7, 7]         2,048
#   ConvBNLayer-52      [[1, 512, 7, 7]]      [1, 512, 7, 7]           0
#      Conv2D-53        [[1, 512, 7, 7]]     [1, 2048, 7, 7]       1,048,576
#   BatchNorm2D-53     [[1, 2048, 7, 7]]     [1, 2048, 7, 7]         8,192
#   ConvBNLayer-53      [[1, 512, 7, 7]]     [1, 2048, 7, 7]           0
# BottleneckBlock-16   [[1, 2048, 7, 7]]     [1, 2048, 7, 7]           0
# AdaptiveAvgPool2D-1  [[1, 2048, 7, 7]]     [1, 2048, 1, 1]           0
#      Linear-1           [[1, 2048]]            [1, 10]            20,490
# ===============================================================================
# Total params: 23,581,642
# Trainable params: 23,475,402
# Non-trainable params: 106,240
# -------------------------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 298.04
# Params size (MB): 89.96
# Estimated Total Size (MB): 388.57
# -------------------------------------------------------------------------------