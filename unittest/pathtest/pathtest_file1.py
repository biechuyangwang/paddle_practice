# 上级引用比较深的下级
from dir1 import dir11_file1 # dir1文件下有__init__.py文件，所以被当做模块(package)
dir11_file1.func()

# 上级调用下级
from dir1 import dir1_file1 # 避免混淆，尽量避免直接导入函数，使用（模块.函数）调用语义更清楚。
# 注意：只有函数名很长，但函数名很有辨识度而且保证不会重名才直接导入函数
dir1_file1.func()

