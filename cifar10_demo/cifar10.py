# 以cifar10来展示深度学习训练及预测流程
# 作者：星期六的故事
# 2022/03/29 08:34


# 加载相关库
import os
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.nn import Layer
from paddle.vision.datasets import Cifar10
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
import numpy as np
from visualdl import LogWriter

# 1 打印环境
print(paddle.__version__)

# 2 查看一下数据，统计一下均值和标准差

# 观察少量数据
# import matplotlib
# import matplotlib.pyplot as plt
# # %matplotlib inline
# train_dataset = Cifar10(mode='train')
# plt.figure()
# idx = 0
# for img, label in train_dataset: # 产看一个数据五个图像
#     if idx==0:
#         print(img)
#         print(label)
#     plt.subplot(1,5,idx+1)
#     plt.imshow(img)
#     idx += 1
#     if idx==5:
#         break
# plt.show()

# train_dataset = Cifar10(mode='train',transform=T.ToTensor())
# test_dataset = Cifar10(mode='test',transform=T.ToTensor())
# print(train_dataset[0])
# print(len(train_dataset))
# print(test_dataset[0])
# print(len(test_dataset))
# means = paddle.zeros([3])
# stds = paddle.zeros([3])
# for img, _ in train_dataset:
#     for d in range(3):
#         means[d] += img[d,:,:].numpy().mean()
#         stds[d] += img[d,:,:].numpy().std()
# means = means.numpy()/len(train_dataset)
# stds = stds.numpy()/len(train_dataset)
# print(means) # [0.491401   0.4821591  0.44653094]
# print(stds) # [0.20220289 0.1993163  0.20086345]

# stats = ((0.491401, 0.4821591, 0.44653094), (0.20220289, 0.1993163, 0.20086345))
# trian_transform = T.Compose([
#     T.RandomCrop(32, padding=4),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     T.Normalize(*stats)
# ])
# test_transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize(*stats)
# ])
# train_dataset = Cifar10(mode='train',transform=trian_transform)
# test_dataset = Cifar10(mode='test',transform=test_transform)
# print(train_dataset[0]) # ( Tensor(shape=[3, 32, 32], dtype=float32), array(6, dtype=int64) ) 
# print(len(train_dataset)) # 50k
# print(test_dataset[0]) # ( Tensor(shape=[3, 32, 32], dtype=float32), array(3, dtype=int64) )
# print(len(test_dataset)) # 10k


# 3 自定义数据集(自己定义数据集) 与上面的构建dataset等价
class MyDateset(paddle.io.Dataset):
    def __init__(self, mode='train'):
        super(MyDateset, self).__init__()

        # 3.1 加载原始数据，并定义数据预处理transform
        stats = ((0.491401, 0.4821591, 0.44653094), (0.20220289, 0.1993163, 0.20086345))
        if mode == 'train':
            self.data = Cifar10(mode='train')
            self.transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*stats)
            ])
        elif mode == 'valid' or mode == 'eval':
            self.data = Cifar10(mode='test')
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(*stats)
            ])
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    
    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = self.data[idx][1]
        image = self.transform(image)
        return image,label

    def __len__(self):
        return len(self.data)

# 4 封装成Dataloader并验证
# train_dataset = MyDateset(mode='train')
# test_dataset = MyDateset(mode='eval')
# print(train_dataset[0]) # ( Tensor(shape=[3, 32, 32], dtype=float32), array(6, dtype=int64) ) 
# print(len(train_dataset)) # 50k
# print(test_dataset[0]) # ( Tensor(shape=[3, 32, 32], dtype=float32), array(3, dtype=int64) )
# print(len(test_dataset)) # 10k

# train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = paddle.io.DataLoader(test_dataset, batch_size=32, shuffle=True)


# 观察一个batch数据
# for img, label in train_loader:
#     print(img.numpy())
#     print(label.numpy())
#     break


# 5 搭建网络模型
class MyModel(Layer):
    def __init__(self,  num_classes=1):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1) # 30*30
        self.relu1 = paddle.nn.ReLU()
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # 15*15

        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1) # 13*13
        self.relu2 = paddle.nn.ReLU()
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # 6*6

        self.conv3 = Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1) # 4*4
        self.relu3 = paddle.nn.ReLU()

        self.linear1 = Linear(in_features=1024, out_features=64)
        self.relu4 = paddle.nn.ReLU()
        self.linear2 = Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)
        return x

# # 在使用GPU机器时，可以将use_gpu变量设置成True
# use_gpu = False
# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# inputs = InputSpec([None, 3*32*32], 'float32', 'x')
# labels = InputSpec([None, 10], 'float32', 'x')
# model = paddle.Model(MyModel(num_classes=10), inputs, labels)

# # 模型可视化
# model.summary((-1,3,32,32))

# 用于模型保存和可视化参数的路径
# filepath, filename = os.path.split(os.path.realpath(__file__))
# stem, suffix = os.path.splitext(filename) # filename .py
# SAVE_DIR = '{}/model/{}'.format(filepath, stem)
# visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))
# print(SAVE_DIR)

# 引入VisualDL库，并设定保存作图数据的文件位置
# from visualdl import LogWriter
# log_writer = LogWriter(logdir='{}/visualdl_log/{}'.format(filepath, stem))

# 6 定义训练流程
def train( # 根据fit定制
    model, 
    train_dataset, 
    test_dataset, 
    optimizer,
    loss,
    metric,
    epochs=1, 
    batch_size=1,
    save_dir=None, # 是否保存模型
    save_freq=1, # 保存频率，单位epoch
    verbose=0, # 是否逐行打印输出
    log_freq=200, # 打印日志的频率
    suffle=True 
    ):

    # 构建dataloader
    if(isinstance(train_dataset,paddle.io.DataLoader)==False):
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=suffle)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size, shuffle=suffle)

    if verbose == 1:
        log_freq = 1
    print('start training ... ')
    # 训练模式
    model.train()

    train_iter = 0
    test_iter = 0
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]

            logits = model(x_data)

            # 计算评价指标
            correct = metric.compute(logits, y_data)
            metric.update(correct)
            acc = metric.accumulate() # 这里是累积的，返回的是平均的acc，格式list
            
            l = loss(logits, y_data)
            avg_loss = paddle.mean(l)

            log_writer.add_scalar(tag = 'train/loss', step = train_iter, value = avg_loss.numpy()[0])
            # log_writer.add_scalar(tag = 'top1 acc', step = iter, value = acc[0])
            # log_writer.add_scalar(tag = 'top5 acc', step = iter, value = acc[1])

            if batch_id % log_freq == 0:
                print("[train] epoch: {}, batch_id: {}, loss is: {:.4f}, top1 acc: {:.4f}, top5 acc: {:.4f}".format(epoch, batch_id, avg_loss.numpy()[0], acc[0], acc[1]))
                # log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                log_writer.add_scalar(tag = 'train/top1_acc', step = train_iter, value = acc[0])
                log_writer.add_scalar(tag = 'train/top5_acc', step = train_iter, value = acc[1])
                metric.reset() # 训练时每输出一次更新一次acc
            l.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_iter += 1
        
        metric.reset() # 避免有累积
        # 每轮后验证一下模型效果
        model.eval() # 修改为评估模式
        losses = []
        for batch_id, data in enumerate(test_loader()):
            x_data = data[0]
            y_data = data[1]

            logits = model(x_data)

            # 计算评价指标
            correct = metric.compute(logits, y_data)
            metric.update(correct)
            acc = metric.accumulate() # 这里是累积的，后面需要平均一下
            
            l = loss(logits, y_data)
            avg_loss = paddle.mean(l)

            losses.append(l.numpy())
            log_writer.add_scalar(tag = 'eval/loss', step = test_iter, value = avg_loss.numpy())
            # log_writer.add_scalar(tag = 'top1 acc', step = iter, value = acc[0])
            # log_writer.add_scalar(tag = 'top5 acc', step = iter, value = acc[1])
            test_iter += 1

        log_writer.add_scalar(tag = 'eval/top1_acc', step = epoch, value = acc[0])
        log_writer.add_scalar(tag = 'eval/top5_acc', step = epoch, value = acc[1])
            
        avg_loss = np.mean(losses)
        # print(avg_loss)
        print("[test] epoch: {}, loss is: {:.4f}, top1 acc: {:.4f}, top5 acc: {:.4f}".format(epoch, avg_loss, acc[0], acc[1]))
        metric.reset() # 避免有累积

        # 保存模型
        if save_dir is not None:
            if epoch+1 == epochs:
                paddle.save(model.state_dict(), '{}/{}.pdparams'.format(SAVE_DIR, 'final'))
                paddle.save(optimizer.state_dict(), '{}/{}.pdopt'.format(SAVE_DIR, 'final'))
                print("epoch {}: Model has been saved in {}.".format('final', SAVE_DIR))
            if epoch % save_freq == 0:
                paddle.save(model.state_dict(), '{}/{}.pdparams'.format(SAVE_DIR, epoch+1))
                paddle.save(optimizer.state_dict(), '{}/{}.pdopt'.format(SAVE_DIR, epoch+1))
                print("epoch {}: Model has been saved in {}.".format(epoch+1, SAVE_DIR))
        model.train() # 转回train模式

# 主函数入口
if __name__ == '__main__':

    # 用于模型保存和可视化参数的路径
    filepath, filename = os.path.split(os.path.realpath(__file__))
    stem, suffix = os.path.splitext(filename) # filename .py
    SAVE_DIR = '{}/model/{}'.format(filepath, stem)
    visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))
    log_writer = LogWriter(logdir='{}/visualdl_log/{}'.format(filepath, stem))
    # print(SAVE_DIR)

    # 加载数据
    train_dataset = MyDateset(mode='train')
    test_dataset = MyDateset(mode='eval')

    # 模型初始化
    model = MyModel(num_classes=10)

    # 定义优化器、损失函数和评价指标
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.01, factor=0.5, patience=5, verbose=True)
    t_optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, momentum=0.9, parameters=model.parameters(), weight_decay=0.001) # Momentum收敛快
    t_loss = paddle.nn.CrossEntropyLoss()
    t_metric = paddle.metric.Accuracy(topk=(1, 5))

    train(model, train_dataset, test_dataset, epochs=3, batch_size=32, optimizer=t_optimizer, loss=t_loss, metric=t_metric, save_dir=SAVE_DIR, save_freq=1)

# visualdl --logdir . --port 8080


# # 恢复训练使用
# # model.load("./mnist_checkpoint/final")

# # 构建模型超参数（优化器，损失，评估准则）
# model.prepare(
#     optim,
#     paddle.nn.CrossEntropyLoss(),
#     Accuracy(topk=(1, 5))
#     )

# # 用于模型保存和可视化参数的路径
# filepath, filename = os.path.split(os.path.realpath(__file__))
# stem, suffix = os.path.splitext(filename) # filename .py
# SAVE_DIR = '{}/model/{}'.format(filepath, stem)
# visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))
# # print(SAVE_DIR)

# # 训练
# model.fit(train_dataset,
#         test_dataset,
#         epochs=10,
#         batch_size=64,
#         save_dir=SAVE_DIR,
#         verbose=1,
#         shuffle=True,
#         callbacks=visualdl
#         )

# # batch_size (int) - 训练数据或评估数据的批大小，当 train_data 或 eval_data 为 DataLoader 的实例时，该参数会被忽略。默认值：1
# # shuffle (bool) - 是否对训练数据进行洗牌。当 train_data 为 DataLoader 的实例时，该参数会被忽略。默认值：True。
# # verbose (int) - 可视化的模型，必须为0，1，2。当设定为0时，不打印日志，
#     # 设定为1时，使用进度条的方式打印日志，设定为2时，一行一行地打印日志。默认值：2。

