# 基于LeNet的mnist分类
# 用于熟悉搭建整个深度学习网络
# 包含：
#   1 自定义数据类：读文件、预处理、封装
#   2 模型设计：网络、损失函数
#   3 自定义训练配置：优化器、资源配置（cpu/gpu）
#   4 自定义训练过程：评价指标、验证集验证、作图、可视化、保存中间结果
#   5 模型保存：为了训练保存/为了预测保存
#   6 部署：动态图转静态图(装饰器 @paddle.jit.to_static)
# 作者：星期六的故事
# 时间：2022/03/25 03:02

# 注意：
#   1 交叉熵内置了softmax，而且针对数值溢出做了优化(每一个输出值减去输出值的最大值)
#   2 To_Tensor()会变化输入的维度，请check维度
#   3 预处理阶段对输入做了正则，预测阶段记得恢复
#   4 构建数据集时，只有样本做了To_Tensor(不作标签是因为预处理样本和标签不能统一做预处理)，训练时记得将标签也做To_Tensor
#   5 
# 加载相关库
import os
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST, FashionMNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
# import paddle.distributed as dist # 用于单机多卡训练
import numpy as np
import matplotlib.pyplot as plt

print(paddle.__version__)

#在使用GPU机器时，可以将use_gpu变量设置成True,可以灵活调整
# print(paddle.device.get_device()) # 查看一下设备
use_gpu = False
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 用于模型保存和可视化参数的路径
filepath, filename = os.path.split(os.path.realpath(__file__))
stem, suffix = os.path.splitext(filename) # filename .py
SAVE_DIR = '{}/model/{}'.format(filepath, stem)
# visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))

# 数据增广
transform = T.Compose([
    T.ToTensor() # 将[H, W, C] 的PIL.Image或者numpy.ndarray转换为[C, H, W]
])

# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode, transform=None):
        datafile = '{}/../dataset/{}'.format(filepath, 'mnist.json.gz') # './work/mnist.json.gz'
        import json
        import gzip
        print('loading mnist dataset from {} ......'.format(datafile))
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        
        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        self.IMG_ROWS = 28
        self.IMG_COLS = 28
        if transform is not None:
            self.transform = transform
        if mode=='train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode=='valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode=='eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
        
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        # 读取图像和标签，转换成需要的尺寸和类型
        img = np.reshape(self.imgs[idx], [self.IMG_ROWS, self.IMG_COLS, 1]).astype('float32') # 保证类似原始图片的[h,w,c]的输入
        label = np.reshape(self.labels[idx], [1]).astype('int64')
        # 数据预处理
        if transform is not None:
            img = self.transform(img)
            # labels = self.transform(labels.numpy())
        return img, label

    def __len__(self):
        return len(self.imgs)
# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
train_dataset = MnistDataset(mode='train', transform=transform)
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
# train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = MnistDataset(mode='valid', transform=transform)
# test_loader = paddle.io.DataLoader(test_dataset, batch_size=64)

# 搭建网络模型
class MyModel(paddle.nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2) 
        # 28*28 卷积核 [Cout, Cin, filter_size_h, filter_size_w],
        # Cout个卷积核 [Cin, H, W]和输入[Cin, H, W]做互相关，得到[H, W]的二维特征（Cin累加了）
        # 卷积层参数w(卷积核的参数) [Cout, Cin, filter_size_h, filter_size_w]

        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # 14*14
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # 10*10
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # 5*5
        self.linear1 = Linear(in_features=16*5*5, out_features=120)
        self.linear2 = Linear(in_features=120, out_features=84)
        self.linear3 = Linear(in_features=84, out_features=10)

    @paddle.jit.to_static  # 添加装饰器，使动态图网络结构在静态图模式下运行
    def forward(self, inputs, label=None, check_shape=False, check_content=False):
        conv1_outputs = self.conv1(inputs)
        x = F.relu(conv1_outputs)
        maxpool1_outputs = self.max_pool1(x)

        conv2_outputs = self.conv2(maxpool1_outputs)
        x = F.relu(conv2_outputs)
        maxpool2_outputs = self.max_pool2(x)

        flatten_outputs = paddle.flatten(maxpool2_outputs, start_axis=1, stop_axis=-1)
        fc1_outputs = self.linear1(flatten_outputs)
        x = F.relu(fc1_outputs)
        fc2_outputs = self.linear2(x)
        x = F.relu(fc2_outputs)
        fc3_outputs = self.linear3(x)


        # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
        if check_shape:
            # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
            print("\n########## print network layer's superparams ##############")
            print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding, self.conv1._stride))
            print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding, self.conv2._stride))
            # print("max_pool1-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool1.pool_size, self.max_pool1.pool_stride, self.max_pool1._stride))
            # print("max_pool2-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool2.weight.shape, self.max_pool2._padding, self.max_pool2._stride))
            print("fc1-- weight_size:{}, bias_size_{}".format(self.linear1.weight.shape, self.linear1.bias.shape))
            print("fc2-- weight_size:{}, bias_size_{}".format(self.linear2.weight.shape, self.linear2.bias.shape))
            print("fc3-- weight_size:{}, bias_size_{}".format(self.linear3.weight.shape, self.linear3.bias.shape))

            # 打印每层的输出尺寸
            print("\n########## print shape of features of every layer ###############")
            print("inputs_shape: {}".format(inputs.shape))
            print("outputs1_shape: {}".format(conv1_outputs.shape))
            print("outputs2_shape: {}".format(maxpool1_outputs.shape))
            print("outputs3_shape: {}".format(conv2_outputs.shape))
            print("outputs4_shape: {}".format(maxpool2_outputs.shape))
            print("outputs5_shape: {}".format(flatten_outputs.shape))
            print("outputs6_shape: {}".format(fc2_outputs.shape))
            print("outputs7_shape: {}".format(fc2_outputs.shape))
            print("outputs8_shape: {}".format(fc3_outputs.shape))
        # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n########## print convolution layer's kernel ###############")
            print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, conv1_outputs.shape[1])
            idx2 = np.random.randint(0, conv2_outputs.shape[1])
            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print("\nThe {}th channel of conv1 layer: ".format(idx1), conv1_outputs[0][idx1])
            print("The {}th channel of conv2 layer: ".format(idx2), conv2_outputs[0][idx2])
            print("The output of last layer:", fc3_outputs[0], '\n')
        # 如果label不是None，则计算分类精度并返回
        if label is not None:
            acc = paddle.metric.accuracy(input=F.softmax(fc3_outputs), label=label)
            return fc3_outputs, acc
        else:
            return fc3_outputs

epoch_num = 10
batch_size = 64
learning_rate = 0.001

#引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter
log_writer = LogWriter(logdir='{}/visualdl_log/{}'.format(filepath, stem))

epoch_num = 10
batch_size = 64
learning_rate = 0.001

val_acc_history = []
val_loss_history = []
def train(model):
    # 修改1- 初始化并行环境
    # dist.init_parallel_env()
    # 修改2- 增加paddle.DataParallel封装
    # model = paddle.DataParallel(model)

    print('start training ... ')
    # turn into training mode
    model.train()

    opt = paddle.optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters()) # weight_decay：L2正则，防止过拟合

    train_loader = paddle.io.DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=batch_size)

    valid_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)

    # 恢复训练使用
    # x_epoch = 'final' # x_epoch可以是任何轮次
    # params_dict = paddle.load('{}/{}.pdparams'.format(SAVE_DIR, x_epoch))
    # opt_dict = paddle.load('{}/{}.pdopt'.format(SAVE_DIR, x_epoch))

    # 加载参数到模型
    # model.set_state_dict(params_dict)
    # opt.set_state_dict(opt_dict)

    iter = 0
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            paddle.to_tensor(y_data)
            # y_data = paddle.to_tensor(data[1])
            # y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            acc = paddle.metric.accuracy(logits, y_data)
            avg_acc = paddle.mean(acc)
            
            loss = F.cross_entropy(logits, y_data)
            avg_loss = paddle.mean(loss)

            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is {}".format(epoch, batch_id, loss.numpy(), avg_acc.numpy()))
                log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
                log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                iter = iter + 200

            loss.backward()
            opt.step()
            opt.clear_grad()

        # 每轮后验证一下模型效果
        model.eval() # 修改为评估模式
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()): # 这里没有验证集，直接用测试集测试，验证的时候是不反传的
            x_data = data[0]
            y_data = data[1]
            paddle.to_tensor(y_data)
            # y_data = paddle.to_tensor(data[1])
            # y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            acc = paddle.metric.accuracy(logits, y_data)
            # logits, acc = model(x_data, y_data) # 如果网络中没有label选项，推荐使用上面的(解耦合)

            loss = F.cross_entropy(logits, y_data)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()
        
        #保存模型参数
        paddle.save(model.state_dict(), '{}/{}.pdparams'.format(SAVE_DIR, epoch))
        paddle.save(opt.state_dict(), '{}/{}.pdopt'.format(SAVE_DIR, epoch))
        print("epoch {}: Model has been saved in {}.".format(epoch, SAVE_DIR))


if __name__ == '__main__':
    model = MyModel()

    # inputs = InputSpec([None, 784], 'float32', 'x')
    # labels = InputSpec([None, 10], 'float32', 'x')
    # model = paddle.Model(MyModel(), inputs, labels)

    # # 模型可视化
    # model.summary((-1,1,28,28))

    # dist.spawn(train) # 默认使用所有可见的gpu
    # dist.spawn(train, nprocs=2) # 使用前2块卡
    # dist.spawn(train, nprocs=2, selelcted_gpus='4,5') # 使用2个进程，使用第4/5号卡
    # python -m paddle.distributed.launch --gpus '0,1' --log_dir ./mylog train.py # 也可以在不改文件，在命令行加参数，推荐，因为可以指定log
    train(model)
    # visualdl --logdir ./mnist_demo/visualdl_log/mnist --port 8080


    # 预测阶段
    # 读取一张本地的样例图片，转变成模型输入的格式
    # def load_image(img_path):
    #     # 从img_path中读取图像，并转为灰度图
    #     im = Image.open(img_path).convert('L')
    #     im = im.resize((28, 28), Image.ANTIALIAS)
    #     im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    #     # 图像归一化
    #     im = 1.0 - im / 255.
    #     return im

    # # 定义预测过程
    # model = MNIST()
    # params_file_path = 'mnist.pdparams'
    # img_path = 'example_0.jpg'

    # # 加载模型参数
    # param_dict = paddle.load(params_file_path)
    # model.load_dict(param_dict)

    # # 灌入数据
    # model.eval()
    # tensor_img = load_image(img_path)
    # #模型反馈10个分类标签的对应概率
    # results = model(paddle.to_tensor(tensor_img)) # 转tensor
    # #取概率最大的标签作为预测输出
    # lab = np.argsort(results.numpy()) # 预测结果需要还原，比如反归一化等
    # print("本次预测的数字是: ", lab[0][-1])

    # 保存推理模型，用于部署
    from paddle.static import InputSpec
    # 加载训练好的模型参数
    x_epoch = 'final'
    state_dict = paddle.load('{}/{}.pdparams'.format(SAVE_DIR, x_epoch))
    # 将训练好的参数读取到网络中
    model.set_state_dict(state_dict)
    # 设置模型为评估模式
    model.eval()

    # 保存inference模型
    paddle.jit.save(
        layer=model,
        path='{}/{}.pdparams'.format(SAVE_DIR, 'inference'), # 保存推理模型(接口)路径
        input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
    print("==>Inference model saved in inference/mnist.")

    # 载入模型用于预测推理或者fine-turn训练
    test_image = None # 来张图片，当然需要做预处理
    # 裁剪、ToTensor
    loaded_model = paddle.jit.load("./inference/mnist")
    preds = loaded_model(test_image)
    pred_label = paddle.argmax(preds)