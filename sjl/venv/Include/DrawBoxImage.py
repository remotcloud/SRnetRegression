import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import _pickle as pickle
sys.path.append('..')
from gplearn._program import _Program
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

if __name__ == '__main__':
    data_dir = 'D:/project/dataset2.0/'
    xlabel = 'K'

    refit_set = []
    for fname in os.listdir(data_dir):
        if fname[-1].isdigit() and fname[0] == 'k':
            with open(f'log/{fname}_program.pkl', 'rb') as f:
                records = pickle.load(f)
            records = records[1]
            refit = []
            for rec in records:
                #refit.append(rec[-1][0][0].raw_fitness_)
                refit.append(rec[0][0].raw_fitness_)
            refit_set.append(refit)
    _draw_fitness_box(f'image/testbox_fit.pdf', refit_set, xlabel='K')
