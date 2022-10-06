"""SrNet in Python, with a scikit-learn and gplearn inspired API.
    Hierarchical fitting neural network using GP with ridge regression,
    and Gp with ridge regression also used to directly fit neural network input and output.
"""

# Author: xilei hu

import sys
import os
sys.path.append('.')
import numpy as np
import pandas as pd
import OneDimenFuncImage as drawwa
import json
import torch
import _pickle as pickle
from data_utils import io,draw
import matplotlib.pyplot as plt
from functions import trans_expression

from ..gplearn.genetic import SymbolicRegressor
from ..gplearn._program import _Program

from joblib import Parallel, delayed
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 创建输入数据  np.newaxis分别是在列(第二维)上增加维度，原先是（300，）变为（300，1）

def save_parameters(param, filepath):
    '''
    保存数据为txt格式

    :param param:
    :param filepath:
    :return:
    '''
    data = param
    if len(param.shape) >= 3:
        data = data.view(data.shape[0], -1)
    np.savetxt(filepath, data)
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

def _draw_f_trend(filename, n_gen, cfs_list, legends):
    mean_fs_list, min_fs_list, max_fs_list = [], [], []
    for cfs in cfs_list:
        mean_fs, min_fs, max_fs = _range_ydatas(cfs, n_gen)
        mean_fs_list.append(mean_fs)
        min_fs_list.append(min_fs)
        max_fs_list.append(max_fs)
    draw.draw_range_line(range(n_gen), min_fs_list, max_fs_list, mean_fs_list,
                         xlabel='gen',
                         ylabel='fitness',
                         legends=legends,
                         savefile=filename)


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


def _range_ydatas(ydatas, x_max):
    mean_ys, min_ys, max_ys = [], [], []
    for i in range(x_max):
        ax_ys = [ydata[i] if i < len(ydata) else ydata[-1] for ydata in ydatas]
        mean_ys.append(np.mean(ax_ys))
        min_ys.append(np.min(ax_ys))
        max_ys.append(np.max(ax_ys))
    return mean_ys, min_ys, max_ys
'''
算出根据gp模型得到的预测值
'''


def predictResult(best_individual, y_preset, y_features, x_features, x_data):
    '''
    :param best_individual: 最好的个体
    :param y_preset: y值预测集合
    :param y_features: y的维数
    :param x_features: x的维数
    :param x_data: 输入自变量
    :return:
    '''
    for m in range(y_features):
        y_pre = 0
        weight = best_individual[1]
        for k in range(x_features):
            x_n = x_data[:, k][:, np.newaxis]  # 每个x的公式对应的x
            exe = best_individual[0][k].execute(x_n)  #f(Xi)
            weight_m_k = weight[m][0][k]
            if type(y_pre).__name__ != 'int':
                if len(y_pre.shape) == 1:
                    y_pre = y_pre[:, np.newaxis]
            if len (exe.shape) ==1:
                exe = exe[:, np.newaxis]
            y_pre = y_pre + weight_m_k * exe  # 计算出每个x的对应公式的值   y = y + Wi*f(Xi)
            #print(y_pre.shape)


        y_pre = y_pre + best_individual[2][m]
        y_pre_shape1 = y_pre.shape[0]
        #print(y_pre.shape)
        y_pre = np.reshape(y_pre, (y_pre_shape1, 1))
        y_preset.append(y_pre)
    all_y_dataPre = y_preset[0]
    for i in range(len(y_preset)):
        if i > 0:
            all_y_dataPre = np.hstack((all_y_dataPre, y_preset[i]))
    return all_y_dataPre


'''
调用训练函数
'''
def trainData(x_data, y_data,extra,populations,gen):
    '''
    :param gen: 代数
    :param populations:种群大小
    :param x_data:输入
    :param y_data: 输出
    :param extra: 外插
    :return:
    '''
    est_gp1 = SymbolicRegressor(population_size=populations,
                                generations=gen, stopping_criteria=0.001,
                                const_range=(-1., 1.),
                                p_crossover=0.9, p_subtree_mutation=0.01,
                                p_hoist_mutation=0.05, p_point_mutation=0.01,
                                metric='mse',
                                max_samples=1.0, verbose=1,
                                parsimony_coefficient=0.01,low_memory = True,
                                random_state=None)

    x_data1 = x_data
    y_data1 = y_data
    est_gp1.fit(x_data1, y_data1)
    #最好的10个个体及其适应度
    best_pro = est_gp1.best_pro_several
    fitset = est_gp1.fitset

    #保存json格式
    '''log_dict = {function_set = function_set,
    arities = arities,
    init_depth = init_depth,
    init_method = init_method,
    n_features =  1,
    metric = metric,
    const_range = const_range,
    p_point_replace = p_point_replace,
    parsimony_coefficient = parsimony_coefficient,
    feature_names = feature_names,
    random_state = random_state,
    program = program}'''


    y_features = y_data1.shape[1]
    x_features = x_data1.shape[1]

    min = np.argmin(fitset)
    y_preSet = []
    best_individual = best_pro[min]
    fitness = fitset[min]
    #内插数据
    y_datapre = predictResult(best_individual, y_preSet, y_features, x_features, x_data1)
    y_preSet = []
    #外插数据
    out_y_dataPre = predictResult(best_individual, y_preSet, y_features, x_features, extra)

    #每代的所有的适应度‘
    fittrend = est_gp1._fittrend
    return best_individual,y_datapre, out_y_dataPre,fitness,fittrend

'''挑选前几个合适的函数训练'''

def trainDataSet(x_data, y_data,extra):
    est_gp1 = SymbolicRegressor(population_size=1500,
                                generations=20, stopping_criteria=0.001,
                                const_range=(-1., 1.),
                                p_crossover=0.9, p_subtree_mutation=0.01,
                                p_hoist_mutation=0.05, p_point_mutation=0.01,
                                metric='mse',
                                max_samples=1.0, verbose=1,
                                parsimony_coefficient=0.0001, random_state=None)

    x_data1 = x_data
    y_data1 = y_data
    est_gp1.fit(x_data1, y_data1)
    fitnessSet = []
    y_features = y_data1.shape[1]
    x_features = x_data1.shape[1]

    y_dataPreSet = []
    out_y_dataPreSet = []
    for i in range(len(est_gp1._best_pro)):
        y_preSet = []
        best_individual = est_gp1._best_pro[i]
        print("The fitness is :",best_individual[0][0].raw_fitness_)
        fitnessSet.append(best_individual[0][0].raw_fitness_)
        #内插数据
        y_datapre = predictResult(best_individual, y_preSet, y_features, x_features, x_data1)
        y_dataPreSet.append(y_datapre)
        y_preSet = []
        #外插数据
        out_y_dataPre = predictResult(best_individual, y_preSet, y_features, x_features, extra)
        out_y_dataPreSet.append(out_y_dataPre)
    return y_dataPreSet, out_y_dataPreSet,fitnessSet
def Parallel_every(data_list,true_data,num_layer,msg,populations,gen) :
    print(msg)
    next_y_datapre = data_list[0]  # 作为下一层的输入
    next_outdata = data_list[0]  # 作为下一层的输出
    srnn_hdata = []     #保存所有隐层的输出值
    best_individualSet, fitnessSet = [], []
    for j in range(num_layer - 1):
        best_individual1, y_dataPre, out_y_dataPre1, fitness1, fitted = trainData(next_y_datapre, data_list[j + 1],
                                                                                  next_outdata,populations,gen)
        # 记录所有的隐层的个体形式、适应度值、预测输出
        best_individualSet.append(best_individual1)
        fitnessSet.append(fitness1)
        srnn_hdata.append(y_dataPre)

        next_y_datapre = y_dataPre
        next_outdata = out_y_dataPre1


        # for i in range(len(y_dataPre)):
    result = {
        'srnn_data':srnn_hdata,
        'best_individual':best_individualSet,
        'fitnessSet':fitnessSet,
        'fitted':fitted
    }
    print(f'{msg} --over!!!')
    return result

    # fittedSet = np.array(fittedSet)

def Parallel_direct(data_list,msg,populaitions,gen):
    # 输入到输出直接训练
    print(msg)
    best_individual2, y_dataPreFinal, out_y_dataPreFinal, fitness2, fitted1 = trainData(data_list[0], data_list[-1],
                                                                                        data_list[0],populaitions,gen)
    result = {
        'sr_data': y_dataPreFinal,
        'sr_best_individual': best_individual2,
        'sr_fitnessSet': fitness2,
        'sr_fitted': fitted1
    }
    return result
    # 计算总体数据

def run(num,num_layer,data_list,true_data,fname,populations,gen):
    '''
    :param num: 运行次数
    :param data_list: 隐层数据
    :param true_data: 真实数据
    :param fname: 函数名
    :return:
    '''

    global best_inDictse, num_bestdict
    '''data = np.loadtxt('../Dataset/f3.txt',dtype=float)
    data = data[data[:,0].argsort()]
    truedata = data[:,1:]'''



    srnn_fitnessSet = []  #神经网络指导的fitness
    srnn_hiddenfit = []
    fittedSet = []      #运行num次程序的所有代的fitness结果
    best_individualSet = []
    srnn_data = []
    srnn_bestfit_of_every = []

    sr_fittedset = []  #
    sr_fitnessSet = []  # 直接符号回归的fitness
    sr_data = []
    sr_best_individualSet = []

    #清空运行文件内容
    #txtIndividul = 'log/'+str(funName)+'/run.txt'
    #file_handle = open(txtIndividul, mode='w',encoding='utf-8')

    #运行10次
    results = Parallel(n_jobs=30)(delayed(Parallel_every)(data_list,true_data,num_layer,f'The {i}th srnn start',populations,gen) for i in range(num))
    for result in results:
        srnn_fitnessSet.append(result['fitnessSet'][-1]) #只保留最后一层
        srnn_hiddenfit.append(result['fitnessSet']) #保留每一层
        fittedSet.append(result['fitted'])
        best_individualSet.append(result['best_individual'])
        srnn_data.append(result['srnn_data'])
    #srnn_fitnessSet = np.array(srnn_fitnessSet)
    #隐层数据
    for data,n in zip(srnn_data,range(num)):
        for hid, k in zip(data,range(len(data))):
            save_parameters(hid, f'./HiddenData/{fname}/{fname}_hidden{k}_num{n}')
    #30次运行适应度
    fittedDic = {}
    for vale,i in zip(fittedSet,range(num)):
        fittedDic[f'list{i}'] = vale
    with open(f'log/fitted/{fname}_30trend.json', 'w') as t:
        json.dump(fittedDic, t, indent=4)

    _draw_f_trend(f'image/{fname}_trend.pdf', gen,[fittedSet], legends=['srnn'])

    #用pickle一个函数的所有运行结果
    fn = f'log/{fname}_program.pkl'
    with open(fn, 'wb') as f:  # open file with write-mode
        picklestring = pickle.dump(best_individualSet, f)  # serialize and save objec

    #使用json30次运行结果的所有个体的保存显示
    num_bestdict = {}
    for best_indiv,entirtyfitness,j in zip(best_individualSet,srnn_hiddenfit,range(num)):

        best_inDictse = {}
        for hidden,fitness_hidden,i in zip(best_indiv,entirtyfitness,range(num_layer-1)):
            express_set = []
            for express in hidden[0]:
                express_set.append( trans_expression(str(express)) )

            best_indivualDict = {
                'fitness' : fitness_hidden,
                'express' : str(express_set),
                'weight' : str(hidden[1]),
                'bias' : str(hidden[2])
            }
            best_inDictse[f'hidden[{i}]'] = best_indivualDict
        num_bestdict[f'individual[{j}]'] =  best_inDictse

    sr_results = Parallel(n_jobs=30)(delayed(Parallel_direct)(data_list,f'The {i}th gp start',populations,gen) for i in range(num))
    for sr_result in sr_results:
        sr_fitnessSet.append(sr_result['sr_fitnessSet'])
        sr_fittedset.append(sr_result['sr_fitted'])
        sr_best_individualSet.append(sr_result['sr_best_individual'])
        sr_data.append(sr_result['sr_data'])
    #sr_fitnessSet = np.array(sr_fitnessSet)
    # 用pickle一个函数的所有运行结果
    fn = f'log/{fname}_sr_program.pkl'
    with open(fn, 'wb') as f:  # open file with write-mode
        picklestring = pickle.dump(sr_best_individualSet, f)  # serialize and save objec

    #使用json30次运行结果的所有个体的保存显示
    sr_num_bestdict = {}
    for hidden,fitness_hidden,j in zip(sr_best_individualSet,sr_fitnessSet,range(num)):

        sr_best_inDictse = {}
        #for hidden,fitness_hidden in zip(sr_best_indiv,sr_entirtyfitness):
        express_set = []
        for express in hidden[0]:
            express_set.append( trans_expression(str(express)) )

        sr_best_indivualDict = {
            'fitness' : fitness_hidden,
            'express' : str(express_set),
            'weight' : str(hidden[1]),
            'bias' : str(hidden[2])
        }

        sr_num_bestdict[f'individual[{j}]'] = sr_best_indivualDict
    #30次运行结果-
    '''sr_fittedDic = {}
    for vale, i in zip(sr_fittedset, range(num)):
        sr_fittedDic[f'list{i}'] = vale
    with open(f'log/sr_fitted/{fname}_30trend.json', 'w') as t:
        json.dump(sr_fittedDic, t, indent=4)'''

    log_dict = {'name': fname,
                'population':populations,
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
                'best_direct_individual':sr_num_bestdict}

    with open(f'log/{fname}_30log.json', 'w') as f:
        json.dump(log_dict, f, indent=4)
    return srnn_fitnessSet
    #file_handle.close()

if __name__ == '__main__':
    data_dir = '/home/huxilei/dataset2.0/'
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
            fitnessSet = run(30,n_layer,data_list,true_data,fname,300,5000)
            dataset_fitness.append(fitnessSet)
    _draw_fitness_box(f'image/box_fit.pdf', dataset_fitness, xlabel=xlabel)
    log_dir = f'image/box_fit.pkl'
    with open(log_dir, 'wb') as f:
        box_fit = pickle.dump(dataset_fitness,f)
