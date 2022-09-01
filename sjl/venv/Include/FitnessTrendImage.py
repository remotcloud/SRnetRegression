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
sys.path.append('..')
from gplearn.genetic import SymbolicRegressor
from gplearn._program import _Program
import matplotlib.pyplot as plt

def _draw_f_trend(filename, n_gen, cfs_list, legends):
    mean_fs_list, min_fs_list, max_fs_list = [], [], []
    for cfs in cfs_list:
        mean_fs, min_fs, max_fs = _range_ydatas(cfs, n_gen)
        mean_fs_list.append(mean_fs)
        min_fs_list.append(min_fs)
        max_fs_list.append(max_fs)
    plt.figure()
    plt.plot(range(5000),mean_fs_list[0])
    plt.show()
    draw.draw_range_line(range(n_gen), min_fs_list, max_fs_list, mean_fs_list,
                         xlabel='gen',
                         ylabel='fitness',
                         legends=legends,
                         savefile=filename)

def draw_f_trend(filename, n_gen, cfs_list, legends=None, title=None, xlabel='gen', ylabel='fitness'):
    n_bar = 9
    interval = (n_gen - n_bar) // n_bar
    mean_fs_list, min_fs_list, max_fs_list = [], [], []
    for cfs in cfs_list:
        mean_fs, min_fs, max_fs = _range_ydatas(cfs, n_gen)
        mean_fs_list.append(mean_fs)
        min_fs_list.append(min_fs)
        max_fs_list.append(max_fs)
    min_fs_list = np.array(min_fs_list)
    max_fs_list = np.array(max_fs_list)
    mean_fs_list = np.array(mean_fs_list)
    draw_error_bar_line(list(range(n_gen)), min_fs_list, max_fs_list, mean_fs_list,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     title=title,
                     legends=legends,
                     savefile=filename,
                     errorevery=(0, interval))
def draw_error_bar_line(x,
                        ys_mins,
                        ys_maxs,
                        ys_means,
                        xlabel,
                        ylabel,
                        title='error bar',
                        legends=None,
                        savefile=None,
                        errorevery=1):
    fig, ax = plt.subplots()

    for y_mins, y_maxs, y_means, l in zip(ys_mins, ys_maxs, ys_means, legends):
        low_err, up_err = y_means-y_mins, y_maxs-y_means
        yerr = np.vstack((low_err, up_err))
        ax.errorbar(x, y_means, yerr=yerr, label=l, ecolor='pink', capsize=2, capthick=1,
                    errorevery=errorevery)
        start_point = str(round(y_means[0].item(), 3))
        end_point = str(round(y_means[-1].item(), 3))
        ax.text(0, y_means[0], start_point)
        ax.text(len(y_means)-1, y_means[-1], end_point)

    if title is not None:
        fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legends is not None and len(legends) > 1:
        plt.legend(loc=2, markerscale=20, bbox_to_anchor=(1.05, 1))
    if savefile:
        plt.savefig(savefile, dpi=600, bbox_inches='tight')
        print(f'saved at {savefile}')
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
draw trend picture
'''
if __name__ == '__main__':
    data_dir = 'D:/project/dataset2.0/'

    for fname in os.listdir(data_dir):
        if fname[-1].isdigit() and fname[0] == 'k':
            n_gen = 5000
            with open(f'log/fitted/{fname}_30trend.json', 'r') as f:
                records = json.load(f)
            data_list = []
            for key,re in records.items():
                data_list.append(list(re))
            '''data_list = np.array(data_list).T
            tdata_list = []'''
            '''for data in data_list:
                for d in data_list:
                    tdata_list.append(list(d))'''
            ylabel = 'fitness'
            draw_f_trend(f'image/{fname}_trend_test.pdf',n_gen,[data_list] ,legends=['srnn'],title=None, ylabel=ylabel)