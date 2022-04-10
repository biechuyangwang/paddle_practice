from urllib import response
import requests
from PIL import Image
from io import BytesIO
from lxml import etree
import time
import random
import os
import re

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


def read_ip_pools(ip_path): # 文件中获取ip池
    ip_list = []
    with open(ip_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip('\n')
            ip_list.append(line)
    return ip_list
# ip_list = read_ip_pools('ip.txt')
# print(ip_list)

import sqlite3
def fetch_db_useful_proxy():
    proxy_list = []
    conn = sqlite3.connect('./database/ip_pools.db')
    # print('ip_pools数据库成功打开')
    cursor = conn.cursor()
    execute_text = 'select proxy from proxys'
    cursor.execute(execute_text)
    res = cursor.fetchall()
    for proxy in res:
        proxy = proxy[0]
        if test_proxies(proxy) == False:
            execute_text = 'delete from proxys where proxy=?'
            cursor.execute(execute_text, (proxy,))
            conn.commit()
            print('{}已删除'.format(proxy))
        else:
            proxy_list.append(proxy)
        # print(proxy[0])
    cursor.close()
    conn.close()
    return proxy_list

def test_proxies(proxy): # 虽然已经筛选过了，但是还是再验证一下
    if re.search(r'^http',proxy) is None:
        proxy = 'http://'+proxy
    proxies = {
        "http":proxy,
        "https":proxy
    }
    print('正在测试：{}'.format(proxies))
    try:
        r = requests.get('https://www.baidu.com',proxies=proxies, timeout=4) # https://pic.netbian.com
        # r = requests.get('https://pic.netbian.com',proxies=proxies, timeout=4)
        # r = requests.get('https://pic.netbian.com', timeout=4)
        if r.status_code == 200:
            print('该代理：{}成功存活'.format(proxy))
            return True
    except:
        print('该代理{}失效!'.format(proxies))
        return False
# ip_list = read_ip_pools('ip.txt')
# print(ip_list)
# for ip in ip_list:
#     test_proxies(ip)

def get_pic(x='4kmeinv',total_page=170,start_page=1):
    ip_list = read_ip_pools('ip.txt')
    
    # headers = {
    #     "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    # }
    # headers = {
    #     "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29"
    # }
    # jar = requests.cookies.RequestsCookieJar()
    # for cookie in cookies.split(";"):
    #     key, value = cookie.split("=", 1)
    #     jar.set(key, value)
    proxy_list = fetch_db_useful_proxy()
    print(proxy_list)
    for idx in range(total_page+1):
        proxy = ""
        if len(proxy_list)>0:
            proxy = random.choice(proxy_list)
        # proxy = random.choice(ip_list)
        # proxy = "http://" + "77.238.129.14:55443" 
        # proxy = "http://" + "183.89.60.236:8080"
        # if idx%2==0:
        #     proxies = {
        #         "http":proxy,
        #         "https":proxy
        #     }
        # else:
        #     proxies = {
        #         "http":"",
        #         "https":""
        #     }
        proxies = {
            "http":proxy,
            "https":proxy
        }
        if idx < start_page: # dongman 94 # shoujibizhi 32
            continue
        if idx == 1:
            url = 'https://pic.netbian.com/{}/index.html'.format(x)
        else:
            url = 'https://pic.netbian.com/{}/index_{}.html'.format(x,idx)
        headers = get_request_headers() # 随机headers
        html = requests.get(url, proxies=proxies, headers=headers, timeout=3)
        cookies = html.cookies
        # print(cookies)
        # break
        html.encoding = 'gbk'
        content = html.text
        # print(content)
        # break
        tree = etree.HTML(content)
        text_list = tree.xpath('//a[contains(@href,"tupian")]/@href')
        for i, text in enumerate(text_list):
            # id = text.split('/')[2].split('.')[0]
            # print(id)
            classid = 54
            root_path = 'https://pic.netbian.com'
            start_url = root_path + text
            # start_url = 'https://pic.netbian.com/downpic.php?id={}&classid={}'.format(id, classid) # 29209 54

            html = requests.get(start_url, proxies=proxies, cookies=cookies, headers=headers, timeout=3)
            html.encoding = 'gbk'
            content = html.text
            tree1 = etree.HTML(content)
            imgs = root_path + tree1.xpath('//img/@src')[0] # /uploads/allimg/180926/092600-15379251603171.jpg
            if re.search(r'uploads',imgs) is None:
                continue
            rootpath = './img/{}/'.format(x)
            filename = tree1.xpath('//img/@title')[0].replace(' ','_').replace('*','x').replace('/','_')
            # filename = re.sub(r'\*', 'x', filename)
            # print(filename)
            filepath = rootpath + filename + '.jpg'

            if os.path.exists(filepath): # 如果已经存在则跳过
                continue
            response1 = requests.get(imgs, proxies=proxies, cookies=cookies, headers=headers, timeout=3)
            if os.path.exists(rootpath) == False:
                os.mkdir(rootpath)
            image = Image.open(BytesIO(response1.content))
            image.save(filepath)
            print(filepath)
            time.sleep(2)
get_pic('4kmeishi',9,1) # type, totalpage, startpage
    # start_url = 'https://pic.netbian.com//downpic.php?id={}&classid={}'.format(id, classid) # 29209 54
    # response1 = requests.get(start_url, cookies=jar, headers=headers, timeout=2)
    # response1.encoding = 'utf-8'
    # print(response1.encoding)
    # filename = response1.headers['Content-Disposition'].split('"')[1].encode('iso-8859-1').decode('gbk').replace(' ', '_')
    # print(filename)
    # print(len(response1.content))
    # filename = 'pic'
    # rootpath = './img/'
    # if os.path.exists(rootpath) == False:
    #     os.mkdir(rootpath)
    # filepath = rootpath + filename + '.jpg'
    # image = Image.open(BytesIO(response1.content))
    # image.save(filepath)
    # print(filepath)


# get_pic('4kmeinv')
# cookies = '__yjs_duid=1_79f4841ab513b593149b9c9c4892cb551649207759738; Hm_lvt_c59f2e992a863c2744e1ba985abaea6c=1649207763; zkhanecookieclassrecord=%2C54%2C; PHPSESSID=bp91thnfff9litm9bb4mg4f3o2; zkhanmlusername=%D0%C7%C6%DA%C1%F9%B5%C4%B9%CA%CA%C2; zkhanmluserid=6056201; zkhanmlgroupid=1; zkhanmlrnd=mf0kQ39GECPacwE4nNWd; zkhanmlauth=a280c36b5bca55031ce6ac2ba8c7d67e; yjs_js_security_passport=8952deca6b32f04c37305a9bd169305095f54f0b_1649415038_js; Hm_lpvt_c59f2e992a863c2744e1ba985abaea6c=1649415200'
# headers = {
#     "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
# }



# for idx,proxy in enumerate(proxy_pools):
#     proxy = 'http://'+proxy_pools[idx]
#     test_proxies(proxy)
# proxy = 'http://'+proxy_pools[0]
# proxies = {
#     "http":proxy,
#     "https":proxy
# }
# # # requests.get(url,proxies=proxies)
# cookies = '__yjs_duid=1_83ee65c4f36701a02b0ee301b4a372ed1649415869645; yjs_js_security_passport=2a1ff27e76232c7a3ec577ee2866d214fab170cf_1649415874_js; Hm_lvt_c59f2e992a863c2744e1ba985abaea6c=1649415874; Hm_lpvt_c59f2e992a863c2744e1ba985abaea6c=1649415874'
# headers = {
#     "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29"
# }

# jar = requests.cookies.RequestsCookieJar()
# url = 'https://pic.netbian.com/tupian/22218.html'
# html = requests.get(url, proxies=proxies, headers=headers, timeout=2)
# html.encoding = 'gbk'
# content = html.text
# print(html)
# tree1 = etree.HTML(content)
# imgs = tree1.xpath('//img/@src')[0]
# print(imgs)
