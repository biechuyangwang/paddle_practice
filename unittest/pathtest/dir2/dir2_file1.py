# 邻居引用
import sys
sys.path.append("C:\\Users\\SAT\\paddle_practice\\unittest\\pathtest") # 将模块所在文件夹加入sys.path后可以正常引用

from dir1 import dir1_file1
dir1_file1.func()