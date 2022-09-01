import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold, datasets

x_data = pd.read_csv("../../Dataset/f5layer1.csv")
x_data = np.array(x_data)

y_data = pd.read_csv("f5_layer1_preData.csv")
y_data = np.array(y_data)
#y_data = y_data[:200]

'''
y_p = []
for m in range(4):
    y_pre=0
    for k in range(4):
        x_n = x_data[:, k][:, np.newaxis]  # 每个x的公式对应的x
        y_pre = y_pre + weight1[m][k] *(0.241/(2*x_n-1.264-1.46*x_n*x_n))# 计算出每个x的对应公式的值
    y_p.append(y_pre)
i = 0
y_pp = y_p[0]
for y_pre in y_p:
    if i > 0:
        y_pp=np.concatenate((y_pp,y_pre),axis=1)
    i=i+1
'''

X = x_data
X2 = y_data

tsne = manifold.TSNE(n_components=2, learning_rate=20, n_iter=1000, perplexity=32, init='random',
                     random_state=90)
X_tsne = tsne.fit_transform(X)
X_tsne1 = tsne.fit_transform(X2)
print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

x_min1, x_max1 = X_tsne1.min(0), X_tsne1.max(0)
X_norm1 = (X_tsne1 - x_min1) / (x_max1 - x_min1)  # 归一化

ax = plt.subplot(1, 2, 1)
for i in range(X_norm.shape[0]):
    ax.text(X_norm[i, 0], X_norm[i, 1], str(2), color=plt.cm.Set1(2),
            fontdict={'weight': 'bold', 'size': 9})
bx = plt.subplot(1, 2, 2)
for i in range(X_norm1.shape[0]):
    bx.text(X_norm1[i, 0], X_norm1[i, 1], str(1), color=plt.cm.Set1(1),
            fontdict={'weight': 'bold', 'size': 9})
plt.show()
