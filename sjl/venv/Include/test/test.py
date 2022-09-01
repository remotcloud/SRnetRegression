import pandas as pd
import numpy as np
import os

x_data = pd.read_csv('aaa.csv', encoding='utf-8')
x_data = np.array(x_data)
y_data = pd.read_csv('../intest.csv', encoding='utf-8')
y_data = np.array(y_data)

x_data=np.concatenate((x_data,y_data),axis=1)
s4 = np.array(x_data)
'''b = pd.DataFrame(s4,columns=["X","Y"])
b.to_csv("dataset.csv",index =0)
data = pd.read_csv('dataset.csv', encoding='utf-8')
with open('xxx.txt', 'a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[0]) +"  "+str(line[1])+ '\n'))'''



'''txt = np.loadtxt('12.txt')
txtDF = pd.DataFrame(txt,columns=["X"])
txtDF.to_csv('gep.csv', index=False)'''
'''txt = np.loadtxt('13.txt')
txtDF = pd.DataFrame(txt,columns=["X"])
txtDF.to_csv('gep1.csv', index=False)
txt = np.loadtxt('14.txt')
txtDF = pd.DataFrame(txt,columns=["X"])
txtDF.to_csv('gep2.csv', index=False)
txt = np.loadtxt('15.txt')
txtDF = pd.DataFrame(txt,columns=["X"])
txtDF.to_csv('gep3.csv', index=False)
txt = np.loadtxt('16.txt')
txtDF = pd.DataFrame(txt,columns=["X"])
txtDF.to_csv('gep4.csv', index=False)
'''
a =np.array([[1],[2]])
b =np.array([[3],[4]])


a=np.concatenate((a,b),axis=1)
a=np.concatenate((a,b),axis=1)
print(a)
