# -*- coding: utf-8 -*-
"""
空白模板
"""
######  欢迎使用脚本任务，首先让我们熟悉脚本任务的一些使用规则  ######
# 详细教程请在AI Studio文档(https://ai.baidu.com/ai-doc/AISTUDIO/Ik3e3g4lt)中进行查看.
# 脚本任务使用流程：
# 1.编写代码/上传本地代码文件
# 2.调整数据集路径以及输出文件路径
# 3.填写启动命令和备注
# 4.提交任务选择运行方式(单机单卡/单机四卡/双机四卡)
# 5.项目详情页查看任务进度及日志
# 注意事项：
# 1.输出结果的体积上限为20GB，超过上限可能导致下载输出失败.
# 2.脚本任务单次任务最大运行时长为72小时（三天）.
# 3.在使用单机四卡或双击四卡时可不配置GPU编号，默认启动所有可见卡；如需配置GPU编号，单机四卡的GPU编号为0,1,2,3；双机四卡的GPU编号为0,1.
# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息.
# -------------------------------关于数据集和输出文件的路径问题---------------------------------
# 数据集路径
# datasets_prefix为数据集的根路径，完整的数据集文件路径是由根路径和相对路径拼接组成的。
# 相对路径获取方式：请在编辑项目状态下通过点击左侧导航「数据集」中文件右侧的【复制】按钮获取.
# datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'
# train_datasets =  datasets_prefix + '通过路径拷贝获取真实数据集文件路径'
# 输出文件路径
# 任务完成后平台会自动把output_dir目录所有文件压缩为tar.gz包，用户可以通过「下载输出」将输出结果下载到本地.
# output_dir = "/root/paddlejob/workspace/output"
# -------------------------------关于启动命令需要注意的问题------------------------------------
# 脚本任务支持两种运行方式
# 1.shell 脚本. 在 run.sh 中编写项目运行时所需的命令，并在启动命令框中填写如 bash run.sh 的命令使脚本任务正常运行.
# 2.python 指令. 在 run.py 编写运行所需的代码，并在启动命令框中填写如 python run.py <参数1> <参数2> 的命令使脚本任务正常运行.
# 注：run.sh、run.py 可使用自己的文件替代，如python train.py 、bash train.sh.
# 命令示例：
# 1. python 指令
# ---------------------------------------单机四卡-------------------------------------------
# 方式一（不配置GPU编号）：python -m paddle.distributed.launch run.py
# 方式二（配置GPU编号）：python -m paddle.distributed.launch --gpus="0,1,2,3" run.py
# ---------------------------------------双机四卡-------------------------------------------
# 方式一（不配置GPU编号）：python -m paddle.distributed.launch run.py
# 方式二（配置GPU编号）：python -m paddle.distributed.launch --gpus="0,1" run.py
# 2. shell 命令
# 使用run.sh或自行创建新的shell文件并在对应的文件中写下需要执行的命令(需要运行多条命令建议使用shell命令的方式)。
# 以单机四卡不配置GPU编号为例，将单机四卡方式一的指令复制在run.sh中，并在启动命令出写出bash run.sh

# 从paddle.vision.models 模块中import 残差网络，VGG网络，LeNet网络
import paddle
from paddle.vision.models import resnet50, vgg16, LeNet
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
import paddle.vision.transforms as T

# 在使用GPU机器时，可以将use_gpu变量设置成True,可以灵活调整
# print(paddle.device.get_device()) # 查看一下设备
# use_gpu = True
# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 用于模型保存和可视化参数的路径
import os
# filepath, filename = os.path.split(os.path.realpath(__file__))
# stem, suffix = os.path.splitext(filename) # filename .py
filepath = './output'
stem = 'cifar10_b' # 版本a表示底层API，版本b表示使用高层API
SAVE_DIR = '{}/model/{}'.format(filepath, stem)
visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))

# 确保从paddle.vision.datasets.Cifar10中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')
# 调用resnet50模型
model = paddle.Model(resnet50(pretrained=True, num_classes=10))

# 使用Cifar10数据集
stats = ((0.491401, 0.4821591, 0.44653094), (0.20220289, 0.1993163, 0.20086345))
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(*stats)
    ])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(*stats)
    ])
train_dataset = Cifar10(mode='train', transform=train_transform)
val_dataset = Cifar10(mode='test', transform=test_transform)
# 定义优化器
optimizer = Momentum(
    learning_rate=0.01, # 应为是高层API这里的学习率没法使用好的策略
    momentum=0.9,
    weight_decay=L2Decay(1e-4),
    parameters=model.parameters())
# 进行训练前准备
model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit( # 如果train_dataset是DataLoader格式的，则batch_size和shuffle失效
    train_dataset,
    val_dataset,
    epochs=100,
    batch_size=64,
    save_dir=SAVE_DIR,
    save_freq=10,
    num_workers=8,
    verbose=1,
    shuffle=True,
    callbacks=visualdl)

# 1. python 指令
# ---------------------------------------单机四卡-------------------------------------------
# 方式一（不配置GPU编号）：python -m paddle.distributed.launch run.py
# 方式二（配置GPU编号）：python -m paddle.distributed.launch --gpus="0,1,2,3" run.py
# ---------------------------------------双机四卡-------------------------------------------
# 方式一（不配置GPU编号）：python -m paddle.distributed.launch run.py
# 方式二（配置GPU编号）：python -m paddle.distributed.launch --gpus="0,1" run.py