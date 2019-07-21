---
layout: mypost
title: Web Crawler
categories: [Python]
---

> 四种爬虫方式

## 正则表达式解析HTML

```
with open('./input/hackdata.html', 'r') as f:
    content = f.read()
pattern =re.compile(r'<p class=\"description\">(.+?)</p>’)     ##正则表达式
course_array = pattern.findall(content)    ##查找所有与pattern匹配的内容
for item in course_array:
    print(item)
```

## lxml解析

```
url_address = requests.get(link)
tree = etree.HTML(url_address.text)
name = tree.xpath('//h1[@itemprop="name"]/text()')[0]
```
	
## BeautifulSoup

```
from bs4 import BeautifulSoup
with open('./input/hackdata.html', 'r') as f:    #指定的解析器
    soup = BeautifulSoup(f, 'html.parser')
    alllines = soup.findAll('p','title')
    for line in alllines:
        content = line.contents[0]
        print(content)
```

## 从API 获取数据
```
import requests
url = 'https://api.douban.com/v2/movie/top250'
params = {'start': 0, 'count':1}
response_info = requests.get(url, params)
print(response_info.url)
print(response_info.status_code)
```