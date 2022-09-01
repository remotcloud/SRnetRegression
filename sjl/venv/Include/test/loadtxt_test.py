import sys

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

loss_set = []
data = np.loadtxt("../Feynman/I.6.2",dtype=float,skiprows=999000)
out = np.loadtxt("../Dataset/f5out.csv",skiprows=999001,dtype=float)
x_data1 = data[:,0:2]
y_data1 = data[:,2:]

z = y_data1
x = x_data1[:,0]
y = x_data1[:,1]

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
bx = fig.add_subplot(212, projection='3d')
'''print('ax.azim {}'.format(ax.azim))  # -60
print('ax.elev {}'.format(ax.elev))  # 30'''
ax.scatter(x, y, z)
bx.scatter(x, y, out)
plt.savefig("1f")
plt.show()
