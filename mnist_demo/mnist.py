# 加载相关库
import os
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST, FashionMNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
import numpy as np
import matplotlib.pyplot as plt

print(paddle.__version__)

# 数据增广
transform = T.Compose([
    T.ToTensor()
])
# 使用transform
print('download training data and load training data')
train_dataset = MNIST(mode='train', transform=transform)
test_dataset = MNIST(mode='test', transform=transform)
print('load finished')

# 搭建网络模型
class MyModel(paddle.nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2) # 28*28
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # 14*14
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # 10*10
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # 5*5
        self.linear1 = Linear(in_features=16*5*5, out_features=120)
        self.linear2 = Linear(in_features=120, out_features=84)
        self.linear3 = Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

epoch_num = 10
batch_size = 64
learning_rate = 0.001

#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()

# 用于模型保存和可视化参数的路径
filepath, filename = os.path.split(os.path.realpath(__file__))
stem, suffix = os.path.splitext(filename) # filename .py
SAVE_DIR = '{}/model/{}'.format(filepath, stem)
# visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))

#引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter
log_writer = LogWriter(logdir='{}/visualdl_log/{}'.format(filepath, stem))

epoch_num = 10
batch_size = 64
learning_rate = 0.001

val_acc_history = []
val_loss_history = []
def train(model):
    print('start training ... ')
    # turn into training mode
    model.train()

    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())

    train_loader = paddle.io.DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=batch_size)

    valid_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)

    iter = 0
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            # y_data = paddle.to_tensor(data[1])
            # y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            acc = paddle.metric.accuracy(logits, y_data)
            avg_acc = paddle.mean(acc)
            
            loss = F.cross_entropy(logits, y_data)
            avg_loss = paddle.mean(loss)

            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
                log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
                log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                iter = iter + 200

            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = data[0]
            y_data = data[1]
            # y_data = paddle.to_tensor(data[1])
            # y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc = paddle.metric.accuracy(logits, y_data)
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

model = MyModel()
train(model)
# visualdl --logdir ./mnist_demo/log --port 8080