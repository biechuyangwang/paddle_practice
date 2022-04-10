"""爬取必应每日背景图并保存在本地

    python bing_crawler.py
"""
import requests
from PIL import Image
from io import BytesIO
import os
    
if __name__ == '__main__':
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    }
    start_url = "http://www.bing.com/HPImageArchive.aspx?format=js&idx=-1&n=10&mkt=zh-cn"
    response1 = requests.get(start_url, headers=headers, timeout=2) # 设置timeout是必要的，防止网络问题导致阻塞
    num = len(response1.json()['images'])
    for i in range(num): # 最大程度时8张
        url = "https://www.bing.com" + response1.json()['images'][i]['urlbase'] + "_UHD.jpg" # 获取图片url
        filename = response1.json()['images'][i]['copyright'].split('(')[0].strip() # 获取图片的名字
        rootpath = './img/bing/' # 图片保存的根目录
        if os.path.exists(rootpath) == False: # 如果根目录不存在则建立一个目录
            os.mkdir(rootpath)
        filepath = rootpath + filename + '.jpg' # 构建图片保存的路径
        if os.path.exists(filepath): # 如果该图片存在则跳过，避免重复请求
            continue
        response2 = requests.get(url, headers=headers, timeout=2)
        image = Image.open(BytesIO(response2.content)) # 从二进制流中读出图片
        image.save(filepath) # 保存
        print(filepath) # 打印调试信息