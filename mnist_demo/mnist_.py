# 加载相关库
import os
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.nn import Layer
from paddle.vision.datasets import MNIST, FashionMNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec

print(paddle.__version__)

# 数据增广
transform = T.Compose([
    T.ToTensor()
])
# 数据的加载和预处理
train_dataset = MNIST(mode='train', transform=transform)
test_dataset = MNIST(mode='test', transform=transform)

# 观察一个数据
# plt.figure()
# img = train_dataset[0][0]
# img = plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)  # 图像二值化展示
# print(train_dataset[0][1])
# plt.show()

# 搭建网络模型
class MyModel(Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2) # 28*28
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

#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()

inputs = InputSpec([None, 784], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(MyModel(), inputs, labels)

# 模型可视化
# model.summary((-1,1,28,28))

optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 恢复训练使用
# params_dict = paddle.load('./checkpoint/mnist_epoch0.pdparams')
# opt_dict = paddle.load('./checkpoint/mnist_epoch0.pdopt')

# 加载参数到模型
# model.set_state_dict(params_dict)
# optim.set_state_dict(opt_dict)
# model.load("./mnist_checkpoint/final")

# 构建模型超参数（优化器，损失，评估准则）
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

# 用于模型保存和可视化参数的路径
filepath, filename = os.path.split(os.path.realpath(__file__))
stem, suffix = os.path.splitext(filename) # filename .py
SAVE_DIR = '{}/model/{}'.format(filepath, stem)
visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))

# print(SAVE_DIR)
# 训练
model.fit(train_dataset,
        test_dataset,
        epochs=5,
        batch_size=64,
        save_dir=SAVE_DIR,
        verbose=1,
        shuffle=True,
        callbacks=visualdl
        )

# import  subprocess
# subprocess.Popen('conda activate autoans', shell=True)
# subprocess.Popen('visualdl --logdir ./mnist_demo/visualdl_log/{}/ --port 8080'.format(filename), shell=True)

# def create_optimizer(parameters):
#     step_each_epoch = len(train_dataset) // BATCH_SIZE
#     # 动态调整学习率
#     lr = paddle.optimizer.lr.CosineAnnealingDecay(
#         learning_rate=0.005, # 初始学习率
#         T_max=step_each_epoch * EPOCHS # 训练轮次的上限
#     )
#     return paddle.optimizer.Momentum(
#         learning_rate = lr,
#         parameters = parameters,
#         weight_decay = paddle.regularizer.L2Decay(0.000001)
#     )

# loss = paddle.nn.CrossEntropyLoss()
# metrics = paddle.metric.Accuracy()

# # 模型参数配置
# model.prepare(
#     optimizer = create_optimizer(net.parameters()),
#     loss = loss,
#     metrics = metrics
# )

# print(net.__class__.__name__)
# 保存模型路径
# SAVE_DIR = './mnist_demo/model/{}/'.format(net.__class__.__name__)
# # 用于可视化的临时参数的路径
# visualdl = paddle.callbacks.VisualDL(log_dir='./mnist_demo/visualdl_log_dir')

# # 启动模型全流程训练
# model.fit(
#     train_dataset,
#     eval_dataset,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     save_dir=SAVE_DIR,
#     verbose=1,
#     shuffle=True,
#     callbacks=[visualdl]
# )

# # 模型评估结果
# predict = model.evaluate(eval_dataset, verbose=1)
# print(predict)

# # 预测
# predict = model.predict(eval_dataset)


# # 测试几组数据用于展示
# # 数字标签对应的实际名称
# list_map = {'0':'t-shirt', '1':'trouser', '2':'pullover', '3':'dress', '4':'coat', '5':'sandal', '6':'shirt', '7':'sneaker', '8':'bag', '9':'ankle boot'}

# # 定义画图方法
# def show_img(img, predict):
#     plt.figure()
#     plt.title('predict: {}'.format(predict))
#     plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)
#     plt.show()

# # 展示样本
# indexs = [0, 16, 64, 256, 1024, 2048, 4096, 8192]

# for idx in indexs:
#     show_img(eval_dataset[idx][0], list_map[str(np.argmax(predict[0][idx]))])

# # 保存用于后续调优的模型
# model.save('finetuning/mnist')

# visualdl --logdir ./mnist_demo/log --port 8080