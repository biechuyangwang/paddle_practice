# 加载飞桨、NumPy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

# 用于模型保存和可视化参数的路径
filepath, filename = os.path.split(os.path.realpath(__file__))
stem, suffix = os.path.splitext(filename) # filename .py
SAVE_DIR = '{}/model/{}'.format(filepath, stem)
# visualdl = paddle.callbacks.VisualDL(log_dir='{}/visualdl_log/{}'.format(filepath, stem))
# print(filepath, filename)
# print(stem, suffix)

#引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter
log_writer = LogWriter(logdir='{}/visualdl_log/{}'.format(filepath, stem))

datafile = '{}/../dataset/{}'.format(filepath, 'housing.data')
def load_data():
    # 从文件导入数据
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset] # 归一化只能使用训练集数据

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# 验证数据集读取程序的正确性
# training_data, test_data = load_data()
# print(training_data.shape)
# print(training_data[1,:])

class MyModel(paddle.nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义模型组件
        self.fc1 = Linear(in_features=13, out_features=13)
        self.fc2 = Linear(in_features=13, out_features=1)

    # 定义前向推理
    def forward(self, inputs):
        x = self.fc2(inputs)
        return x

model = MyModel()
model.train()
training_data, test_data = load_data()

opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

EPOCH_NUM = 500   # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
test_mini_batches = [test_data[k:k+BATCH_SIZE] for k in range(0, len(test_data), BATCH_SIZE)]
val_acc_history = []
val_loss_history = []
iter = 0
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    train_mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for batch_id, mini_batch in enumerate(train_mini_batches):
        x = np.array(mini_batch[:, :-1], dtype='float32') # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:], dtype='float32') # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor的格式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)
        
        # 前向计算
        predicts = model(house_features)

        # 计算acc
        # acc = paddle.metric.accuracy(predicts, prices)
        # avg_acc = paddle.mean(acc)

        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)


        # 对结果做反归一化处理
        predicts = predicts * (max_values[-1] - min_values[-1]) + avg_values[-1]
        # 对label数据做反归一化处理
        prices = prices * (max_values[-1] - min_values[-1]) + avg_values[-1]
        squared_err = paddle.abs(predicts-prices)/prices
        avg_squared_err = paddle.mean(squared_err)

        if batch_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            log_writer.add_scalar(tag = 'squared_err', step = iter, value = avg_squared_err.numpy())
            log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
            iter = iter + 20
        
        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()
    
    # evaluate model after one epoch
    model.eval()
    squared_errs = []
    losses = []
    for batch_id, mini_batch in enumerate(test_mini_batches):
        x = np.array(mini_batch[:, :-1]) # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:]) # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor的格式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 前向计算
        predicts = model(house_features)

        # 计算acc,这里不需要计算mini-batch的均值了，直接累积算整个epoch的
        # acc = paddle.metric.accuracy(predicts, prices)

        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)

        # 对结果做反归一化处理
        predicts = predicts * (max_values[-1] - min_values[-1]) + avg_values[-1]
        # 对label数据做反归一化处理
        prices = prices * (max_values[-1] - min_values[-1]) + avg_values[-1]
        squared_err = paddle.abs(predicts-prices)/prices
        squared_errs.extend(squared_err.numpy()) # extend会将元素逐个加入列表
        # print(squared_errs[:1])

        # accuracies.append(acc.numpy())
        losses.extend(loss.numpy())

    avg_squared_err, avg_loss = np.mean(squared_errs), np.mean(losses)
    print("[validation] squared_err/loss: {}/{}".format(avg_squared_err, avg_loss))
    # print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
    val_acc_history.append(avg_squared_err)
    val_loss_history.append(avg_loss)
    model.train()
    #保存模型参数
    if epoch_id%50==0:
        paddle.save(model.state_dict(), '{}/{}.pdparams'.format(SAVE_DIR, epoch_id))
        paddle.save(opt.state_dict(), '{}/{}.pdopt'.format(SAVE_DIR, epoch_id))
        print("epoch {}: Model has been saved in {}.".format(epoch_id, SAVE_DIR))


# def load_one_example():
#     # 从上边已加载的测试集中，随机选择一条作为测试数据
#     idx = np.random.randint(0, test_data.shape[0])
#     idx = -10
#     one_data, label = test_data[idx, :-1], test_data[idx, -1]
#     # 修改该条数据shape为[1,13]
#     one_data =  one_data.reshape([1,-1])

#     return one_data, label

# model.eval()
# # 参数为数据集的文件地址
# one_data, label = load_one_example()
# # 将数据转为动态图的variable格式 
# one_data = paddle.to_tensor(one_data)
# predict = model(one_data)

# # 对结果做反归一化处理
# predict = predict * (max_values[-1] - min_values[-1]) + avg_values[-1]
# # 对label数据做反归一化处理
# label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]

# print('predict/label: {}/{}'.format(predict, label))