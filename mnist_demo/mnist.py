# 加载相关库
import os
import random
import paddle
import paddle.fluid as fluid
from paddle.vision.transforms import Compose, Normalize
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')
# for i in range(len(train_dataset)):
#     sample = train_dataset[i]
#     print(sample[0].size, sample[1])
#     break

# 定义数据集读取器
def load_data(mode='train',BATCH_SIZE=100):

    # 读取数据文件
    # train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=True)
    # datafile = './work/mnist.json.gz'
    # print('loading mnist dataset from {} ......'.format(datafile))
    # data = json.load(gzip.open(datafile))
    # # 读取数据集中的训练集，验证集和测试集
    # train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_dataset
    elif mode == 'eval':
        imgs = test_dataset
    # 获得所有图像的数量
    imgs_length = len(imgs)

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = BATCH_SIZE

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i][0], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(imgs[i][1], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

# 定义模型结构
class MNIST_MODEL(fluid.dygraph.Layer):
     def __init__(self):
         super(MNIST_MODEL, self).__init__()
         
         # 定义一个卷积层，使用relu激活函数
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个卷积层，使用relu激活函数
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个全连接层，输出节点数为10 
         self.fc = Linear(input_dim=980, output_dim=10, act='softmax')
     
     #加入对每一层输入和输出的尺寸和数据内容的打印，根据check参数决策是否打印每层的参数和输出尺寸
     def forward(self, inputs, label=None, check_shape=False, check_content=False):
         # 给不同层的输出不同命名，方便调试
         outputs1 = self.conv1(inputs)
         outputs2 = self.pool1(outputs1)
         outputs3 = self.conv2(outputs2)
         outputs4 = self.pool2(outputs3)
         _outputs4 = fluid.layers.reshape(outputs4, [outputs4.shape[0], -1])
         outputs5 = self.fc(_outputs4)
         
         # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
         if check_shape:
             # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
             print("\n########## print network layer's superparams ##############")
             print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding, self.conv1._stride))
             print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding, self.conv2._stride))
             print("pool1-- pool_type:{}, pool_size:{}, pool_stride:{}".format(self.pool1._pool_type, self.pool1._pool_size, self.pool1._pool_stride))
             print("pool2-- pool_type:{}, poo2_size:{}, pool_stride:{}".format(self.pool2._pool_type, self.pool2._pool_size, self.pool2._pool_stride))
             print("fc-- weight_size:{}, bias_size_{}, activation:{}".format(self.fc.weight.shape, self.fc.bias.shape, self.fc._act))
             
             # 打印每层的输出尺寸
             print("\n########## print shape of features of every layer ###############")
             print("inputs_shape: {}".format(inputs.shape))
             print("outputs1_shape: {}".format(outputs1.shape))
             print("outputs2_shape: {}".format(outputs2.shape))
             print("outputs3_shape: {}".format(outputs3.shape))
             print("outputs4_shape: {}".format(outputs4.shape))
             print("outputs5_shape: {}".format(outputs5.shape))
             
         # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
         if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
             print("\n########## print convolution layer's kernel ###############")
             print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
             print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

             # 创建随机数，随机打印某一个通道的输出值
             idx1 = np.random.randint(0, outputs1.shape[1])
             idx2 = np.random.randint(0, outputs3.shape[1])
             # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
             print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
             print("The {}th channel of conv2 layer: ".format(idx2), outputs3[0][idx2])
             print("The output of last layer:", outputs5[0], '\n')
            
        # 如果label不是None，则计算分类精度并返回
         if label is not None:
             acc = fluid.layers.accuracy(input=outputs5, label=label)
             return outputs5, acc
         else:
             return outputs5

#调用加载数据的函数
BATCH_SIZE = 100
train_loader = load_data('train',BATCH_SIZE=BATCH_SIZE)
params_path = "./mnist_demo/mnist"

#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

#引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter
log_writer = LogWriter(logdir="./mnist_demo/log")

with fluid.dygraph.guard():
    params_dict, opt_dict = fluid.load_dygraph(params_path)
    model = MNIST_MODEL()
    model.train() 
    
    EPOCH_NUM = 10
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(60000//BATCH_SIZE) + 1) * EPOCH_NUM
    lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.001)

    #四种优化算法的设置方案，可以逐一尝试效果
    # optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    #optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.01, momentum=0.9, parameter_list=model.parameters())
    #optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    #optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    
    #各种优化算法均可以加入正则化项，避免过拟合，参数regularization_coeff调节正则化项的权重
    #optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model.parameters()))
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1),
                                              parameter_list=model.parameters())
    
    
    iter = 0
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            # if batch_id == 0 and epoch_id==0:
            #     # 打印模型参数和每层输出的尺寸
            #     predict, avg_acc = model(image, label, check_shape=True, check_content=False)
            # elif batch_id==401:
            #     # 打印模型参数和每层输出的值
            #     predict, avg_acc = model(image, label, check_shape=False, check_content=True)
            # else:
            #     predict, avg_acc = model(image, label)
            
            predict, avg_acc = model(image, label)
            avg_acc = fluid.layers.mean(avg_acc)
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
                log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                iter = iter + 100
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

        #保存模型参数
        fluid.save_dygraph(model.state_dict(), './mnist_demo/checkpoint/mnist_epoch{}'.format(epoch_id))
        fluid.save_dygraph(optimizer.state_dict(), './mnist_demo/checkpoint/mnist_epoch{}'.format(epoch_id))
        print("epoch {}: Model has been saved.".format(epoch_id))

# visualdl --logdir ./mnist_demo/log --port 8080