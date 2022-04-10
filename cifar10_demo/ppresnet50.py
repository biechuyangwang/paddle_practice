# 从paddle.vision.models 模块中import 残差网络，VGG网络，LeNet网络
import paddle
from paddle.vision.models import resnet50, vgg16, LeNet
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.transforms import Transpose

#在使用GPU机器时，可以将use_gpu变量设置成True,可以灵活调整
# print(paddle.device.get_device()) # 查看一下设备
use_gpu = False
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 用于模型保存和可视化参数的路径
import os
filepath, filename = os.path.split(os.path.realpath(__file__))
stem, suffix = os.path.splitext(filename) # filename .py
SAVE_DIR = '{}/model/{}'.format(filepath, stem)
visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))

# 确保从paddle.vision.datasets.Cifar10中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')
# 调用resnet50模型
model = paddle.Model(resnet50(pretrained=False, num_classes=10))

# 使用Cifar10数据集
train_dataset = Cifar10(mode='train', transform=Transpose())
val_dataset = Cifar10(mode='test', transform=Transpose())
# 定义优化器
optimizer = Momentum(
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=L2Decay(1e-4),
    parameters=model.parameters())
# 进行训练前准备
model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit(
    train_dataset,
    val_dataset,
    epochs=50,
    batch_size=64,
    save_dir=SAVE_DIR,
    num_workers=8,
    verbose=1,
    shuffle=True,
    callbacks=visualdl)