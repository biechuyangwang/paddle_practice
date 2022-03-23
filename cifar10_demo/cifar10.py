# 加载相关库
import os
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.nn import Layer
from paddle.vision.datasets import Cifar10
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec

print(paddle.__version__)

# 数据增广
transform = T.Compose([
    T.Transpose(),
    T.Normalize([127.5], [127.5])
])
# 数据的加载和预处理
train_dataset = Cifar10(mode='train', transform=transform)
test_dataset = Cifar10(mode='test', transform=transform)

# 观察一个数据
# plt.figure()
# img = train_dataset[0][0]
# img = plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)  # 图像二值化展示
# print(train_dataset[0][1])
# plt.show()

# 搭建网络模型
class MyModel(Layer):
    def __init__(self,  num_classes=1):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1) # 30*30
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # 15*15

        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1) # 13*13
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # 6*6

        self.conv3 = Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1) # 4*4

        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()

inputs = InputSpec([None, 3*32*32], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(MyModel(num_classes=10), inputs, labels)

# 模型可视化
# model.summary((-1,3,32,32))

optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 恢复训练使用
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
        epochs=10,
        batch_size=32,
        save_dir=SAVE_DIR,
        verbose=1,
        shuffle=True,
        callbacks=visualdl
        )

# batch_size (int) - 训练数据或评估数据的批大小，当 train_data 或 eval_data 为 DataLoader 的实例时，该参数会被忽略。默认值：1
# shuffle (bool) - 是否对训练数据进行洗牌。当 train_data 为 DataLoader 的实例时，该参数会被忽略。默认值：True。
# verbose (int) - 可视化的模型，必须为0，1，2。当设定为0时，不打印日志，
    # 设定为1时，使用进度条的方式打印日志，设定为2时，一行一行地打印日志。默认值：2。

