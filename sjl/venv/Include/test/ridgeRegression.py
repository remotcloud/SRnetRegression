from sklearn.linear_model import Ridge
from sklearn import datasets,linear_model

import sklearn.linear_model
import numpy as np
import sys
import os
import traceback

import sympy

sys.path.append('..')
import numpy as np
import pandas as pd
import drawing as drawwa
import json
import torch
import _pickle as pickle
from data_utils import io, draw
import matplotlib.pyplot as plt

def _get_datanames(n):
    """if your datalist is named as like ['input', 'hidden1', 'hidden2', ..., 'output']
    then you can use this method for convenience."""
    if n < 2:
        raise ValueError("n should >= 2")
    names = []
    for i in range(n):
        if i == 0:
            names.append('input')
        elif i == n - 1:
            names.append('output')
        else:
            names.append(f'hidden{i}')
    return names

data_dir = 'D:/project/dataset2.0/'
xlabel = 'K'

dataset_fitness = []
for fname in os.listdir(data_dir):
    if fname[-1].isdigit() and fname[-2] == 'k':
        nn_dir = f'{data_dir}{fname}_nn/'
        true_file = f'{data_dir}{fname}'
        n_layer = 0

        for hfile in os.listdir(nn_dir):
            if hfile.startswith('hidden') or hfile == 'input' or hfile == 'output':
                n_layer += 1
        data_names = _get_datanames(n_layer)
        data_list = io.get_datalist(nn_dir, data_names)
        true_data = io.get_dataset(true_file)

        # n_samples, n_features = 10, 1
        # rng = np.random.RandomState(0)
        # y = rng.randn(n_samples)
        # X = rng.randn(n_samples, n_features)
        clf = Ridge()
        lnr = linear_model.LinearRegression()
        las = linear_model.Lasso(alpha=0.01)
        las.fit(data_list[0],true_data)
        lnr.fit(data_list[0],true_data)
        clf.fit(data_list[0],true_data)
        #print(clf.score(X,y))
        print(y)
        print(clf.predict(X))
        print(lnr.predict(X))
        print(las.predict(X))
        print(clf.coef_)
        print(lnr.coef_)
        print(las.coef_)
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = las.fit(X, y)
reg.score(X, y)
print(reg.predict(np.array([[3, 5]])))
