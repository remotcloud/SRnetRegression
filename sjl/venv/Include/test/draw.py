import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''y_true =pd.read_csv("f3out.csv")
y_true = np.array(y_true)



x_data = pd.read_csv("f3input.csv")
x_data = np.array(x_data)


plt.figure()

lines = plt.plot(x_data, y_true, 'b-', lw=3,label='network neutral')

plt.legend()
plt.show()'''

'''ax1 = plt.axes(projection='3d')
z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

n = 200
x = y = np.linspace(-10, 10, n)
z = np.random.randn(n)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print('ax.azim {}'.format(ax.azim))  # -60
print('ax.elev {}'.format(ax.elev))  # 30

# elev, azim = 0, 0
# ax.view_init(elev, azim)  # 设定视角

ax.scatter(x, y, z)

plt.show()
