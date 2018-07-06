
# coding: utf-8

# In[1]:

#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

#read the file
Test1 = pd.read_excel("TEJ_Excel練習_台積電.xlsx")


# In[3]:

#head() shows the top five piece of data
Test1.head()


# In[4]:

#take a look at the type of each cloumn
Test1.dtypes


# In[5]:

#choose the column of time and Closing_Price
#time = Test1['年月日']
#CP = Test1['收盤價']


# In[6]:

#take a look at "time"
t#ime.head()


# In[7]:

#Check the data structure of "time"
#type(time)


# In[8]:

#trans object to datetime for drawing a picture
#Time = pd.to_datetime(time)


# In[9]:

#ensure the data structure is datetime
#Time.head()


# In[10]:

#take a look at closing price; It's float. 
#CP.head()


# In[ ]:




# In[11]:

#from time import strftime, strptime


# In[12]:

#String
#time[1]


# In[13]:

#string to structure_time
#t2 = strptime(time[1],"%Y/%m/%d")#字串依照年月日切開放到t2


# In[14]:

#seperate into categories
#t2


# In[15]:

#call the content from t2 and store in "a" as string type.
#a = strftime("%Y %m %d",t2)


# In[16]:

#check it
#a


# In[27]:

ax = plt.subplot(111)

t1 = np.arange(0.0, 1.0, 0.01)

for n in [1, 2, 3, 4]:
    plt.plot(t1, t1**n, label="n=%d"%(n,))

#新增x軸標籤,旋轉45度可以避免卡在一起
plt.xticks([0,0.2,0.4,0.7,0.8,1],rotation = 45)
#圖例
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)

#圖例背景透明度
leg.get_frame().set_alpha(0.5)


plt.show()


# In[157]:

#定義函數
def f1(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

def f2(t):
    return np.sin(2*np.pi*t)*np.cos(3*np.pi*t)
#資料輸入數值
t = np.arange(0.0,5,0.02)

#設定畫布 figsize(左右寬度,上下長度), dpi 視窗大小
plt.figure(figsize=(10,7),dpi=100)
#分配上下兩圖 subplot(column row index)
p1 = plt.subplot(211)#(2columnwidth 1rowlength index1)
p2 = plt.subplot(212)#(2columnwidth 1rowlength index2)

#數學式$式子$
label_f1 = "$f(t)=e^{-t} \cos (2 \pi t)$"
label_f2 = "$g(t)=\sin (2 \pi t) \cos (3 \pi t)$"

#圖案主體
p1.plot(t,f1(t),"g-",label=label_f1)
p2.plot(t,f2(t), color = "#F98BB0", label=label_f2, lw=5)

#設定p1 x,y軸上下界刻度
p1.axis([0.0,5.01,-1.0,1.5])

#p1 x,y軸標題
p1.set_xlabel("$x$", fontsize=14, color = "#103900")
p1.set_ylabel("$y$", fontsize=14, color = "#103900")

#P1圖形標題,fontisize字體大小
p1.set_title(label_f1,fontsize=18, loc ="right", color = "#103900")
#格線
p1.grid(True)
#圖例
#p1.legend()

#p1註解標籤位置
tx = 2
ty = 0.9
#p1.text(tx,ty,label_f1,fontsize=15,verticalalignment="top",horizontalalignment="right")
#https://matplotlib.org/users/text_props.html

#最大值標籤, axis setting
max_y = max(f1(t)) 
max_x = t[np.where(max(f1(t)))]
p1.text(max_x,max_y+0.35,"$max$",fontsize= 15,verticalalignment="top",horizontalalignment="left")
#最大值箭頭設定
p1.annotate('',xy=(max_x, max_y),xytext=(max_x +0.2, max_y + 0.2),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

#最小標籤
min_y = min(f1(t))
min_x = t[np.argmin(f1(t))]
p1.text(min_x-0.1,min_y-0.2,"$min$", fontsize = 15, verticalalignment = "top", horizontalalignment = "right")
p1.annotate('',xy=(min_x, min_y),xytext=(min_x -0.2, min_y - 0.2),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))


p2.axis([0.0,5.01,-1.0,1.5])
p2.set_ylabel("$y$",fontsize=14, color = "#F92A6F")
p2.set_xlabel("$x$",fontsize=14, color = "#F92A6F")
p2.set_title(label_f2,fontsize=18, loc = "right", color = "#F98BB0")
#p2.legend()

tx = 2
ty = 0.9
p2.text(tx,ty,label_f2,fontsize=15,verticalalignment="bottom",horizontalalignment="right")
p2.annotate('',xy=(1.8,0.5),xytext=(tx,ty),arrowprops=dict(facecolor="#F98BB0", shrink=0.01))

plt.show()

#plt.savefig('arrow.png', transparent = True, bbox_inches = 'tight', pad_inches = 0.25) 


# In[144]:

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.1, 0.1)
y5 = [1.0,0.4118,0.325,0.3182,0.2769,0.2688,0.2403,0.2378,0.2222,0.565,0.00]
y6 = [0.8571,0.8571,0.75,0.7368,0.6216,0.6216,0.551,0.3636,0.1722,0.0,0.0]
y7 = [0.5,0.2208,0.2208,0.2208,0.1957,0.1929,0.1929,0.1448,0.0905,0.0421,0.00]
y8 = [1.0,0.9,0.9,0.7778,0.6333,0.6053,0.5745,0.2324,0.18,0.0695,0.00]
name = ["Okapi TF","Mle_Laplace","Jelinek-Mercer","Dirichlet"]
plt.plot(t1, y5, label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = "#FCE302", marker ="o", mew = 10, mec = "#FC9402")
plt.plot(t1, y6, label="method = %s"%(name[1]), drawstyle = 'steps-post', lw = 5, color = "#0CF42B", marker ="^", mew = 6, mec = "#08AF1F")
plt.plot(t1, y7, label="method = %s"%(name[2]), drawstyle = 'steps-post', lw = 1, color = "#FF0077", marker ="s", mew = 4, mec = "#FF0000")
plt.plot(t1, y8, label="method = %s"%(name[3]), drawstyle = 'steps-post', lw = 7, color = "#2FB6F9", marker =".", mew = 10, mec = "#392FF9")
plt.title('$Un-interpolated recall precision average (withstop)$', color = "#392FF9",fontsize=15)

plt.plot()
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)
#背景格線
plt.grid(True)

leg = plt.legend(loc='upper right', bbox_to_anchor=(1,1),ncol=1, shadow=True, fancybox=True)
leg.get_frame().set_alpha(1)


plt.show()


# In[ ]:

#parameter
name = ["Okapi TF","Mle_Laplace","Jelinek-Mercer","Dirichlet"]
setting = [221, 222, 223, 224]#num(row),num(column),index
linecolor = ["#FCE302", "#0CF42B", "#FF0077", "#2FB6F9"]


# In[159]:

#picture
fig = plt.figure()
fig.add_subplot(221)   #top left(221)/(2,2,1)
plt.plot(Time[:30], CP[:30], label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = "#2FB6F9", marker ="o", mew = 1, mec = "#392FF9")
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('年月日', color = "#392FF9",fontsize=15)
plt.grid(True)

fig.add_subplot(222)   #top left(221)/(2,2,1)
plt.plot(Time[30:60], CP[30:60], label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = "#2FB6F9", marker ="o", mew = 1, mec = "#392FF9")
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)
plt.grid(True)

fig.add_subplot(223)   #top left(221)/(2,2,1)
plt.plot(Time[60:90], CP[60:90], label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = "#2FB6F9", marker ="o", mew = 1, mec = "#392FF9")
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)
plt.grid(True)

fig.add_subplot(224)   #top left(221)/(2,2,1)
plt.plot(Time[90:120], CP[90:120], label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = "#2FB6F9", marker ="o", mew = 1, mec = "#392FF9")
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)
plt.grid(True)
plt.show()
plt.tight_layout()


# In[63]:

#picture
fig = plt.figure()
fig.add_subplot(111)#projection='polar',axisbg='r'
plt.plot(Time[90:120], CP[90:120], label="method = %s"%(name[0]), lw = 7, color = "#2FB6F9")
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)
plt.show()


# In[ ]:




# In[108]:

#polar
#開一張圖
fig = plt.figure()
r = np.arange(0, 2, 0.01)
r1 = np.arange(2, 0, -0.01)

#開一張極座標圖,設定背景顏色
fig.add_subplot(111, projection='polar',axisbg="#E2F9B8")
plt.plot(2 * np.pi * r, r,color = "#A5F8D3",lw =5)
plt.plot(2 * np.pi * r, r1,color = "#D16014",lw =5)
plt.yticks([0.4, 1, 1.5, 2],color = "#D16014",)
plt.show()


# In[3]:

#parameter
t1 = np.arange(0.0, 1.1, 0.1)
y1 = [1.0,0.4118,0.325,0.3182,0.2769,0.2688,0.2403,0.2378,0.2222,0.565,0.00]
y2 = [0.8571,0.8571,0.75,0.7368,0.6216,0.6216,0.551,0.3636,0.1722,0.0,0.0]
y3 = [0.5,0.2208,0.2208,0.2208,0.1957,0.1929,0.1929,0.1448,0.0905,0.0421,0.00]
y4 = [1.0,0.9,0.9,0.7778,0.6333,0.6053,0.5745,0.2324,0.18,0.0695,0.00]
yy = [y1,y2,y3,y4]
name = ["Okapi TF","Mle_Laplace","Jelinek-Mercer","Dirichlet"]
setting = [221, 222, 223, 224]#num(row),num(column),index
linecolor = ["#FCE302", "#0CF42B", "#FF0077", "#2FB6F9"]


# In[ ]:

#draw a picture
fig = plt.figure()
for i in range(len(yy)):
    fig.add_subplot(setting[i])
    plt.plot(t1,yy[i],label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = linecolor[i], marker ="o", mew = 10, mec = "#FC9402")


# In[4]:

#picture
fig = plt.figure()
#plt.grid(True)
fig.add_subplot(221)   #top left(221)/(2,2,1)
plt.plot(t1, y5, label="method = %s"%(name[0]), drawstyle = 'steps-post', lw = 3, color = "#FCE302", marker ="o", mew = 10, mec = "#FC9402")
#X軸
plt.xlabel('年月日', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)
plt.grid(True)


fig.add_subplot(222)   #top right
plt.plot(t1, y6, label="method = %s"%(name[1]), drawstyle = 'steps-post', lw = 5, color = "#0CF42B", marker ="^", mew = 6, mec = "#08AF1F")
plt.grid(True)
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)


fig.add_subplot(223)   #bottom left
plt.plot(t1, y7, label="method = %s"%(name[2]), drawstyle = 'steps-post', lw = 1, color = "#FF0077", marker ="s", mew = 4, mec = "#FF0000")
plt.grid(True)
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)


fig.add_subplot(224)   #bottom right 
plt.plot(t1, y8, label="method = %s"%(name[3]), drawstyle = 'steps-post', lw = 7, color = "#2FB6F9", marker =".", mew = 10, mec = "#392FF9")
plt.grid(True)
#X軸
plt.xlabel('Recall', color = "#392FF9",fontsize=15)
#Y軸
plt.ylabel('Precision', color = "#392FF9",fontsize=15)

plt.show()


# In[6]:

ax = plt.subplot(111)
X = [0,1,2,3]
name = np.array(["Okapi TF","Mle_Laplace","Jelinek-Mercer","Dirichlet"])

NStop = [0.2629,0.3136,0.1541,0.5123]
Stop = [0.2629,0.4668,0.1541,0.5123]
plt.bar(X, Stop, fc='#FF758E', ec='none',align="center")
plt.xticks((0,1,2,3),(name[0],name[1],name[2],name[3]),color ="#392FF9",fontsize=15)  
plt.title('Un-interpolated mean average precision (withstop)', color = "#392FF9",fontsize=15) 
#plt.xlabel('Precision', color = "#FF0077")
for a,b in zip(X,Stop):
    plt.text(a, b+0.005, '%f' % b, ha='center', va= 'bottom',fontsize=15,color = '#33223D')
    print(b)
plt.ylabel('準度', color = "#392FF9",fontsize=15)
plt.show()


# In[43]:

plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()


# In[62]:

import calendar


# In[63]:

calendar.month(2018,6)


# In[64]:

print(calendar.month(2018,6))


# In[50]:

fig = plt.figure()
fig.add_subplot(3, 1, 1)   #row,column,index
fig.add_subplot(3, 1, 2)   
fig.add_subplot(3, 1, 3)  
plt.show()


# In[ ]:



