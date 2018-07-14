
# coding: utf-8

# 方法一
# 1.執行 plt.rcParams['font.sans-serif'] = ['SimHei']
# 2.將C:\Windows\Fonts</span>內部的$simhei.ttf$複製到Anaconda3中安裝matplotlib下mpl-data中的fonts
# E.g.C:\Users\名\Anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf
# (方法一速度快，缺點為必須每次都多執行一行程式碼)

# 方法二
# A.
# 1.到Windows系統字體資料夾中，找一個喜歡的中文字體檔案，e.g.simhei.ttf
# 2.將Matplotlib中default的字體檔案Vera.ttf，所選的中文字體重新命名為Vera.ttf
# 
# B.
# 打開mpl-data資料夾中的matplotlibrc檔案，可以用Vim或記事本打開
# 
# 1.修改中文字體顯示設定:將font.family的值改為SimHei，並將前面的#去除；將font.serif中的取值新增SimHei
#     >font.family:SimHei
#     >中間程式碼省略
#     >font.sans-serif:SimHei, Bitstream Vera Sans, Lucida Grande, Verdana...
# 2.將axes.unicode_minus True改為False
#     >axes.unicode_minus:False
# C.
# 修改rcsetup.py
# 1.打開rcsetup.py
#     
#     >a map from key -> value, converter<br>>defaultParams = {
#     >省略
#     >'font.family':[[sans-serif], validate_stringlist]
#     >省略
#     >'font.sans-serif':[['SimHei','Bitstream Vera Sans', 'DejaVu Sans', 'Lucida Granda, 'Verdana'...]]
#     >}
# 
# 2.一樣將axes.unicode_minus True改為False
#     ># a map from key -> value, converter
#     >defaultParams = {
#     >省略
#     >'axes.unicode_minus':[False, validate_bool],
#     >省略
#     >}
# 
# 完成A,B and C後就可以正常顯示中文了，不必額外執行程式碼

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:

#讀檔
df = pd.read_excel("TEJ_Excel練習_台積電.xlsx")


# In[3]:

#看columns
df.columns


# In[4]:

#切資料
closing_price = df.iloc[:,2]
opening_price = df.iloc[:,3]


# In[5]:

plt.figure(figsize=(10,7),dpi=100)
#分配上下兩圖 subplot(column row index)
p1 = plt.subplot(211)#(2columnwidth 1rowlength index1)
p2 = plt.subplot(212)#(2columnwidth 1rowlength index2)

#圖案主體
p1.plot(closing_price, color = "#F4247E", alpha = 0.6, lw=5)
p2.plot(opening_price, color = "#16A080", lw=5)

#背景格線
p1.grid(True, axis='y', ls='--', color='r', alpha=0.8)
p2.grid(True, axis='y', ls='--', color='g', alpha=0.8)

#正常顯示中、英文
p1.set_title('收盤價', color="#CE0059", fontsize = 20)
p2.set_title('$Opening$ $price$', color="#128469", fontsize = 20)

#y軸標籤
p1.set_ylabel('股價', color="#CE0059", fontsize = 20)
p2.set_ylabel('$Stock$ $Price$', color="#128469", fontsize = 20)
plt.show()

