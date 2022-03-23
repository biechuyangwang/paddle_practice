import numpy as np
import paddle

input_1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
index_1 = np.array([0,1])
input = paddle.to_tensor(input_1)
index = paddle.to_tensor(index_1)
output = paddle.gather(input, index, axis=1)
print(output)
# expected output: [[1,2],[4,5],[7,8]]

# paddle.gather
# 根据索引index获取输入x的指定aixs维度的条目，并将它们拼接在一起。