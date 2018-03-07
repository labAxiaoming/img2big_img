
使用一系列皮肤图片组合构成王者荣耀的logo

原理比较简单 每个方块利用总像素差最小原则来匹配

## 1先运行get_skin.py来爬取图片 会自己建一个文件夹，图片存储在文件夹下


```
#爬虫代码get_skin.p
import urllib.request
import json
import os

response = urllib.request.urlopen('http://pvp.qq.com/web201605/js/herolist.json') 

hero_json = json.loads(response.read())

hero_num = len(hero_json)

save_dir = 'heroskin\\'  #文件夹创建

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
for i in range(hero_num):
    # 获取英雄皮肤列表
    skin_names = hero_json[i]['skin_name'].split('|')
    
    for cnt in range(len(skin_names)):
        save_file_name = save_dir + str(hero_json[i]['ename']) + '-' +hero_json[i]['cname']+ '-' +skin_names[cnt] + '.jpg'
        skin_url = 'http://game.gtimg.cn/images/yxzj/img201606/skin/hero-info/'+str(hero_json[i]['ename'])+ '/' +str(hero_json[i]['ename'])+'-bigskin-' + str(cnt+1) +'.jpg'
        if not os.path.exists(save_file_name):
            urllib.request.urlretrieve(skin_url, save_file_name)

#爬虫代码结束
```



bgg.jpg是大图，也可以自己定义
![background](https://github.com/labAxiaoming/img2big_img/blob/master/bgg.jpg)



## 2运行.py得到大图

![goal picture](https://github.com/labAxiaoming/img2big_img/blob/master/bgg3.jpg)
