"""爬取ip池

    python ip_crawler.py
"""
import requests
from lxml import etree
from PIL import Image
from io import BytesIO
import os

def test_proxies(proxy): # 验证有效性
    proxies = {
        "http":proxy,
        "https":proxy
    }
    print('正在测试：{}'.format(proxies))
    try:
        r = requests.get('https://www.baidu.com',proxies=proxies, timeout=3)
        if r.status_code == 200:
            print('该代理：{}成功存活'.format(proxy))
            return True
    except:
        print('该代理{}失效!'.format(proxies))
        return False

def save_ip_pools(ip_pools):
    # res = []
    file = open('ip.txt','a')
    for ip in ip_pools:
        if test_proxies(ip):
            # res.append(ip)
            file.write(ip+'\n')
    file.close()

if __name__ == '__main__':

    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    }
    ids = [333205,333206,333207]
    urls = []
    for id in ids:
        url = 'https://www.zdaye.com/dayProxy/ip/{}.html'.format(id)
        urls.append(url)
        url = 'https://www.zdaye.com/dayProxy/ip/{}/2.html'.format(id)
        urls.append(url)
    # id = 333206
    # url = 'https://www.zdaye.com/dayProxy/ip/{}.html'.format(id)
    # url = 'https://www.zdaye.com/dayProxy/ip/{}/2.html'.format(id)
    
    if os.path.exists('ip.txt'):
        os.remove('ip.txt')
    for url in urls:
        html = requests.get(url,headers=headers,timeout=2)
        html.encoding = 'gbk'
        content = html.text
        # print(content)
        tree1 = etree.HTML(content)
        ip_list = tree1.xpath('//a[contains(@href,"CheckHttp")]/@href')
        ip_list = ['http://' + ip.split('/')[3] for ip in ip_list if ip.split('/')[3] != '']
        # print(ip_list)
        save_ip_pools(ip_list)

    