import sys
import os
from data_utils import io
import traceback

import sympy
from sympy import *
sys.path.append('.')
import numpy as np

import OneDimenFuncImage as drawwa
import json
import torch
import _pickle as pickle
from data_utils import io, draw
import matplotlib.pyplot as plt
from functions import trans_expression

sys.path.append('..')
from gplearn.genetic import SymbolicRegressor
from gplearn_original.genetic import SymbolicRegressor as OSR
from gplearn._program import _Program

from joblib import Parallel, delayed

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
def _draw_fitness_box(filename, srnn_fs_list, xlabel=None):
    if not xlabel:
        xlabels = [f'F{i}' for i in range(len(srnn_fs_list))]
    else:
        xlabels = [f'{xlabel}{i}' for i in range(len(srnn_fs_list))]

    fig, ax = plt.subplots()
    ax.boxplot(srnn_fs_list, vert=True, patch_artist=True, labels=xlabels)
    ax.set_xlabel('Problem')
    ax.set_ylabel('Fitness')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
def Parallel_every(data_list, true_data, num_layer, msg, populations, gen):
    print(msg)
    next_y_datapre = data_list[0]  # 作为下一层的输入
    next_outdata = data_list[0]  # 作为下一层的输出
    srnn_hdata = []  # 保存所有隐层的输出值
    best_individualSet, fitnessSet = [], []
    for j in range(num_layer - 1):
        best_individual1, y_dataPre, out_y_dataPre1, fitness1, fitted = trainData(next_y_datapre, data_list[j + 1],
                                                                                  next_outdata, populations, gen)
        # 记录所有的隐层的个体形式、适应度值、预测输出
        best_individualSet.append(best_individual1)
        fitnessSet.append(fitness1)
        srnn_hdata.append(y_dataPre)

        next_y_datapre = y_dataPre
        next_outdata = out_y_dataPre1

        # for i in range(len(y_dataPre)):
    result = {
        'srnn_data': srnn_hdata,
        'best_individual': best_individualSet,
        'fitnessSet': fitnessSet,
        'fitted': fitted
    }
    print(f'{msg} --over!!!')
    return result

    #
def run(num, num_layer, data_list, true_data, fname, populations, gen):
    '''
    :param num: 运行次数
    :param data_list: 隐层数据
    :param true_data: 真实数据
    :param fname: 函数名
    :return:
    '''

    global best_inDictse, num_bestdict
    srnn_fitnessSet = []  # 神经网络指导的fitness
    srnn_hiddenfit = []
    fittedSet = []  # 运行num次程序的所有代的fitness结果
    best_individualSet = []
    srnn_data = []
    srnn_bestfit_of_every = []

    sr_fittedset = []  #
    sr_fitnessSet = []  # 直接符号回归的fitness
    sr_data = []
    sr_best_individualSet = []
    # 运行10次
    results = Parallel(n_jobs=4)(
        delayed(Parallel_every)(data_list, true_data, num_layer, f'The {fname} {i}th srnn start', populations, gen) for i in
        range(num))
    for result in results:
        srnn_fitnessSet.append(result['fitnessSet'][-1])  # 只保留最后一层
        srnn_hiddenfit.append(result['fitnessSet'])  # 保留每一层
        fittedSet.append(result['fitted'])
        best_individualSet.append(result['best_individual'])
        srnn_data.append(result['srnn_data'])
    # srnn_fitnessSet = np.array(srnn_fitnessSet)
    # 隐层数据
    for data, n in zip(srnn_data, range(num)):
        for hid, k in zip(data, range(len(data))):
            io.save_parameters(hid, f'./HiddenData/{fname}/{fname}_hidden{k}_num{n}')
    # 30次运行适应度
    fittedDic = {}
    for vale, i in zip(fittedSet, range(num)):
        fittedDic[f'list{i}'] = vale
    with open(f'log/fitted/{fname}_30trend.json', 'w') as t:
        json.dump(fittedDic, t, indent=4)
    ylabel = 'fitness'

    # 使用json30次运行结果的所有个体的保存显示
    num_bestdict = {}
    express_final = []

    # 用pickle保存一个函数的所有运行结果
    final_result = [best_individualSet,express_final]
    fn = f'log/{fname}_program.pkl'
    with open(fn, 'wb') as f:  # open file with write-mode
        picklestring = pickle.dump(final_result, f)  # serialize and save objec


    # run result
    log_dict = {'name': fname,
                'population': populations,
                'gen': gen,
                'srnn_mean_fitness': str(np.mean(srnn_fitnessSet)),
                'sr_mean_fitness': str(np.mean(sr_fitnessSet)),
                'srnn_fitness': srnn_fitnessSet,
                'srnn_min_fitness': np.min(srnn_fitnessSet),
                'srnn_max_fitness': np.max(srnn_fitnessSet),
                'sr_fitness': sr_fitnessSet,
                'sr_min_fitness': np.min(sr_fitnessSet),
                'sr_max_fitness': np.max(sr_fitnessSet),
                'best_individual': num_bestdict,
                'best_direct_individual': sr_num_bestdict}

    with open(f'log/{fname}_30log.json', 'w') as f:
        json.dump(log_dict, f, indent=4)
    return srnn_fitnessSet

if __name__ == '__main__':
    data_dir = '../Dataset/'
    xlabel = 'F'

    dataset_fitness = []
    for fname in os.listdir(data_dir):
        if fname[-1].isdigit() and fname[-2] == 'k':
            nn_dir = f'{data_dir}{fname}/'
            true_file = f'{data_dir}{fname}'
            n_layer = 0

            for hfile in os.listdir(nn_dir):
                if hfile.startswith('hidden') or hfile == 'input' or hfile == 'output':
                    n_layer += 1
            data_names = _get_datanames(n_layer)
            data_list = io.get_datalist(nn_dir, data_names)
            true_data = io.get_dataset(true_file)
            fitnessSet = run(4, n_layer, data_list, true_data, fname, 2000, 200)
            dataset_fitness.append(fitnessSet)
    _draw_fitness_box(f'image/box_fit.pdf', dataset_fitness, xlabel=xlabel)
    log_dir = f'image/box_fit.pkl'
    with open(log_dir, 'wb') as f:
        box_fit = pickle.dump(dataset_fitness, f)