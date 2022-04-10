"""验证ip池

    python ip_valid.py
"""
import requests
from lxml import etree
import re
from PIL import Image
from io import BytesIO
import os
import sqlite3

def check_proxy_valida(proxy): # 验证有效性
    if re.search(r'^http',proxy) is None:
        proxy = 'http://'+proxy
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

def read_ip_pools(ip_path):
    proxy_list = []
    with open(ip_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip('\n')
            proxy_list.append(line)
    return proxy_list

def db_valid():
    conn = sqlite3.connect('./database/ip_pools.db')
    # print('ip_pools数据库成功打开')
    cursor = conn.cursor()
    execute_text = 'select proxy from proxys'
    cursor.execute(execute_text)
    res = cursor.fetchall()
    for proxy in res:
        proxy = proxy[0]
        if check_proxy_valida(proxy) == False:
            execute_text = 'delete from proxys where proxy=?'
            cursor.execute(execute_text, (proxy,))
            conn.commit()
            print('{}已删除'.format(proxy))
        # print(proxy[0])
    cursor.close()
    conn.close()
# db_valid()

if __name__ == '__main__':
    conn = sqlite3.connect('./database/ip_pools.db')
    # print('ip_pools数据库成功打开')
    cursor = conn.cursor()
    useful_proxy = []
    proxy_list = read_ip_pools('ip.txt')
    for proxy in proxy_list:
        if check_proxy_valida(proxy):
            useful_proxy.append(proxy)
            execute_text = 'select proxy from proxys where proxy=?'
            cursor.execute(execute_text,(proxy,))
            res = cursor.fetchone()
            if res is None:
                execute_text = 'INSERT INTO proxys(proxy, score) VALUES(?, ?)'
                content = (proxy, 5)
                cursor.execute(execute_text, content)
                conn.commit()
    cursor.close()
    conn.close()
    print(useful_proxy)
    db_valid() # 验证一遍数据库

# import sqlite3

# conn = sqlite3.connect('./database/ip_pools.db')
# print('ip_pools数据库成功打开')
# cur = conn.cursor()
# cur.execute('''create table proxys \
#     (id INTEGER PRIMARY KEY AUTOINCREMENT, \
#     proxy char(50) NOT NULL UNIQUE, \
#     score INT NOT NULL \
#     );''')
# print('proxys数据表创建成功')
# conn.commit()
# cur.close()
# conn.close()