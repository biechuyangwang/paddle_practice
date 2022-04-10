"""爬取必应每日背景图并保存在本地

    python 4Kpic_crawler.py
"""
import requests
import re
import sys
from PIL import Image
import time
import random
from io import BytesIO
import os

# 用户代理User-Agent列表
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
    "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
    "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
    "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
    "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
    "UCWEB7.0.2.37/28/999",
    "NOKIA5700/ UCWEB7.0.2.37/28/999",
    "Openwave/ UCWEB7.0.2.37/28/999",
    "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
    # iPhone 6：
    "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25",
]


#随机获取一个用户代理User-Agent的请求头
def get_request_headers():
	headers = {
	'User-Agent':random.choice(USER_AGENTS),
	'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
	'Accept-language':'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer':'https://www.baidu.com',
	'Accept-Encoding':'gzip, deflate,br',
	'Connection':'keep-alive',
	}
	return headers

def get_pic(cid, start=0, count=10):
    # url_list = []
    rootpath = './img/{}/'.format(cid)
    if os.path.exists(rootpath) == False:
        os.mkdir(rootpath)

    headers = get_request_headers()
    start_url = 'https://api.xiaobaibk.com/lab/wallpaper/api.php?cid={}&start={}&count={}'.format(cid, start, count)
    response1 = requests.get(start_url, headers=headers, timeout=2) # 设置timeout是必要的，防止网络问题导致阻塞
    # print(cookies)
    # break
    cookies = None
    response1.encoding = 'gbk'
    num = len(response1.json()['data'])
    for i in range(num): 
        r1 = "[a-zA-Z0-9\_]+"
        url = 'http:' + response1.json()['data'][i]['url'].split(':')[1]
        tag = re.sub('[\s]+','_',re.sub(r1, '', response1.json()['data'][i]['tag'])).replace('_美女模特','').replace('全部_','')
        # utag = response1.json()['data'][i]['utag'].replace(' ','_')
        # url_list.append(url)
        # print('{},{},{}'.format(i,url,utag))
        filename = tag
        filepath = rootpath + '{}_'.format(start+i) +filename + '.jpg'

        if os.path.exists(filepath): # 如果已经存在则跳过
            continue
        if i==0:
            response2 = requests.get(url, headers=headers, timeout=3)
            cookies = response1.cookies
        else:
            response2 = requests.get(url, cookies=cookies, headers=headers, timeout=3)
        try:
            image = Image.open(BytesIO(response2.content)) # 像这种文件打开的，最好有异常处理
            image.save(filepath)
            # print(tag)
            print(filepath)
            time.sleep(1)
        except:
            continue

# get_pic(6,0,2)

if __name__ == '__main__':
    cid = 6
    if len(sys.argv)>=2:
        cid = sys.argv[1]
    start_epoch = 200
    if len(sys.argv)>=3:
        start_epoch = int(sys.argv[2])
    end_epoch = 370
    if len(sys.argv)>=4:
        end_epoch = int(sys.argv[3])
    iter_num = 20
    for x in range(start_epoch,end_epoch):
        print(x)
        get_pic(cid, x * iter_num, iter_num) # 每轮iter_num个



    # headers = {
    #     "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    # }
    # start_url = "https://cn.bing.com/HPImageArchive.aspx?format=js&idx=0&n=8"
    # response1 = requests.get(start_url, headers=headers, timeout=2) # 设置timeout是必要的，防止网络问题导致阻塞
    # for i in range(8): # 最大程度时8张
    #     url = "https://www.bing.com" + response1.json()['images'][i]['urlbase'] + "_UHD.jpg" # 获取图片url
    #     filename = response1.json()['images'][i]['copyright'].split('(')[0].strip() # 获取图片的名字
    #     rootpath = './img/' # 图片保存的根目录
    #     if os.path.exists(rootpath) == False: # 如果根目录不存在则建立一个目录
    #         os.mkdir(rootpath)
    #     filepath = rootpath + filename + '.jpg' # 构建图片保存的路径
    #     if os.path.exists(filepath): # 如果该图片存在则跳过，避免重复请求
    #         continue
    #     response2 = requests.get(url, headers=headers, timeout=2)
    #     image = Image.open(BytesIO(response2.content)) # 从二进制流中读出图片
    #     image.save(filepath) # 保存
    #     print(filepath) # 打印调试信息