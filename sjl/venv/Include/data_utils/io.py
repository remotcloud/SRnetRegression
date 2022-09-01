import torch
import pickle
import numpy as np

import os


# return a tensor dataset
def get_dataset(filepath):
    dataset = np.loadtxt(filepath)
    if len(dataset.shape) == 1:
        dataset = dataset[:,np.newaxis]
    return dataset


def get_datalist(file_dir, nameList):
    dataList = []
    for i in range(len(nameList)):
        dataList.append(get_dataset(file_dir+nameList[i]))
    return dataList


# report hyper parameters to file
def save_report(message, filepath, mode='a'):
    with open(filepath, mode) as f:
        f.write(message+'\n')


# save layer's parameters to file
def save_parameters(param, filepath):
    data = param
    if len(param.shape) >= 3:
        data = data.view(data.shape[0], -1)
    np.savetxt(filepath, data)


def save_layers(layers, save_dir):
    for i in range(len(layers)):
        if i == 0:
            name = 'input'
        elif i == len(layers)-1:
            name = 'output'
        else:
            name = 'hidden%d' % (i)
        np.savetxt(save_dir+name, layers[i])


def save_nn_model(nn, savepath):
    torch.save(nn, savepath)


def load_nn_model(nnpath):
    nn = torch.load(nnpath)
    nn.eval()
    return nn


def save_objs(objs, obj_names, savedir):
    for obj, name in zip(objs, obj_names):
        with open(f'{savedir}{name}.p', 'wb') as f:
            pickle.dump(obj, f)


def load_objs(obj_names, dir):
    objs = []
    for name in obj_names:
        with open(f'{dir}{name}.p', 'rb') as f:
            objs.append(pickle.load(f))

    return objs


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)