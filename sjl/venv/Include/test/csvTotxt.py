import sys


import numpy as np
import pandas as pd
def save_parameters(param, filepath):
    data = param
    if len(param.shape) >= 3:
        data = data.view(data.shape[0], -1)
    np.savetxt(filepath, data)


inp = pd.read_csv("../f1input.csv")
fint = np.array(inp)
layer1 = pd.read_csv("../f1layer1.csv")
layer1 = np.array(layer1)
layer2 = pd.read_csv("../f1layer2.csv")
layer2 = np.array(layer2)
layer3 = pd.read_csv("../f1layer3.csv")
layer3 = np.array(layer3)
otest = pd.read_csv("../f1outtest.csv")
otest = np.array(otest)

save_parameters(layer1,"D:/Dataset/tf1layer1")
save_parameters(layer2,"D:/Dataset/tf1layer2")
save_parameters(layer3,"D:/Dataset/tf1layer3")