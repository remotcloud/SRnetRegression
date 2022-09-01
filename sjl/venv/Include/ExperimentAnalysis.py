import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import _pickle as pickle
import torch

from nn_models import NN_MAP
from kkkfunc import _func_map, _Function,_range_map
from data_utils import io, draw

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

def draw_hidden_compare_img_version2(filename,run_num, nndatas, srnndatas):
    def normalize(_datas):
        _data_maxs, _data_mins = [], []
        for _data in _datas:
            _data_maxs.append(np.max(_data))
            _data_mins.append(np.min(_data))
        _data_max, _data_min = max(_data_maxs), min(_data_mins)
        for _i, _data in enumerate(_datas):
            _datas[_i] = (_data - _data_min) / (_data_max - _data_min)
        return _datas

    n_row, n_col = 2, 2
    n_samples = n_row * n_col

    n_h = len(nndatas)
    space = nndatas[0].shape[0] // n_samples
    idxs = [i*space for i in range(n_samples)]

    nn_samples = [np.vstack([nndatas[i][j] for j in idxs]) for i in range(n_h)]
    srnn_samples = [np.vstack([srnndatas[i][j] for j in idxs]) for i in range(n_h)]
    nn_norms = normalize(nn_samples)
    srnn_norms = normalize(srnn_samples)

    for h_idx, hs in enumerate(zip(nn_norms, srnn_norms)):
        nnh, srnnh = hs
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, sharex=True, sharey=True)
        for i, ax in enumerate(axes.flat):
            grid = np.vstack((nnh[i], srnnh[i]))
            # YlGn or summer would be great
            im = ax.imshow(grid, cmap='summer', vmax=1, vmin=0)
            for h in range(2):
                [ax.text(j, h, round(grid[h, j].item(), 1), ha='center', va='center', color='b') for j in range(grid.shape[1])]

        fig.subplots_adjust(hspace=0)
        fig.suptitle(f'h{h_idx}')

        cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        if(filename):
            plt.savefig(f'{filename}_h{h_idx}_r{run_num}.pdf', dpi=300)
            print(f'saved at {filename}_h{h_idx}_r{run_num}')
        plt.show()
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
def predictResult(best_individual, y_preset, y_features, x_features, x_data):
    '''
    求出个体相对于x_data的预测输出
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
def load_nn_model(nnpath, load_type='dict', nn=None):
    if load_type == 'model':
        nn = torch.load(nnpath)
    elif load_type == 'dict':
        if not nn:
            raise ValueError(f'karg nn should not be None when you specify load_type as dict')
        nn.load_state_dict(torch.load(nnpath))
    else:
        raise ValueError(f'karg load_type should be one of model or dict, not {load_type}.')
    nn.eval()
    return nn

def image_compare(data_dir):
    entdata_dir = data_dir
    for fname in os.listdir(entdata_dir):

        for run_num in range(4):
            srnn_data_dir = './HiddenData/'
            #fname = 'kkk5'

            nn_dir = f'{entdata_dir}{fname}_nn/'
            true_file = f'{entdata_dir}{fname}'
            n_layer = 0

            for hfile in os.listdir(nn_dir):
                if hfile.startswith('hidden') or hfile == 'input' or hfile == 'output':
                    n_layer += 1
            data_names = _get_datanames(n_layer)
            nn_datas = io.get_datalist(nn_dir, data_names)
            true_data = io.get_dataset(true_file)

            #run_num = 15
            srnn_datas = []
            for n in range(n_layer-1):
                hidden = f'{fname}/{fname}_hidden{n}_num{run_num}'
                srnn = np.loadtxt(srnn_data_dir + hidden)
                srnn_datas.append(srnn)

            #对隐层进行对比
            img_dir = f'./image/compare/'
            draw_hidden_compare_img_version2(f'{img_dir}{fname}',run_num,
                                             nn_datas[1:-1],
                                             srnn_datas[:-1])


            #读取对象计算值
            log_dir = f'log/{fname}_program.pkl'
            with open(log_dir, 'rb') as f:
                individual_Set = pickle.load(f)
            individual_Set = individual_Set[0]
            #重新画图
            range_func = _range_map[f'{fname}']
            kkk_func = _func_map[f'{fname}']
            arg = []
            y0 = np.random.uniform(2*range_func[0], 2*range_func[1], 2*range_func[2])[:, np.newaxis]
            #y0 = np.linspace(2*range_func[0], 2*range_func[1], 2*range_func[2])[:, np.newaxis]
            for num in range(kkk_func.arity):
                arg.append(y0)
            x_data = arg[0]
            flag = 0
            for a in arg:
                if flag > 0:
                    x_data = np.hstack((x_data,a))
                flag = flag + 1
            #true data
            t_data = kkk_func(*arg)
            y_datapre = x_data
            for hidden_num in range(n_layer-1):
                x_features = nn_datas[hidden_num].shape[1]
                y_features = nn_datas[hidden_num+1].shape[1]
                y_preSet = []
                y_datapre = predictResult(individual_Set[run_num][hidden_num], y_preSet, y_features, x_features,y_datapre)

            #network

            nn=load_nn_model(f'D:/project/dataset2.0/{fname}_nn/nn_module.pt',nn=NN_MAP[f'{fname}']).cpu()
            x_data = torch.from_numpy(x_data).float()
            nn_predict = list(nn(x_data))[-1]
            nn_predict = nn_predict.detach().numpy()

            #画图
            plt.figure()
            plt.title(f'{fname}_{run_num}_out')
            plt.scatter(y0, y_datapre, label='sr',)
            plt.scatter(y0, t_data, label='true')
            plt.scatter(y0, nn_predict, label='nn')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=10, bbox_to_anchor=(1.05, 1))
            plt.savefig(f'image/ts/{fname}_output_{run_num}.pdf', dpi=600, bbox_inches='tight')
            plt.show()
def image_compareData3(data_dir):
    entdata_dir = data_dir
    for fname in os.listdir(entdata_dir):

        for run_num in range(1):
            srnn_data_dir = './HiddenData/'
            #fname = 'kkk5'

            nn_dir = f'{entdata_dir}{fname}/'
            true_file = f'{entdata_dir}{fname}/{fname}.txt'
            n_layer = 0

            for hfile in os.listdir(nn_dir):
                if hfile.startswith('hidden') or hfile == 'input' or hfile == 'output':
                    n_layer += 1
            data_names = _get_datanames(n_layer)
            nn_datas = io.get_datalist(nn_dir, data_names)
            true_data = io.get_dataset(true_file)

            #run_num = 15
            srnn_datas = []
            for n in range(n_layer-1):
                hidden = f'{fname}/{fname}_hidden{n}_num{run_num}'
                srnn = np.loadtxt(srnn_data_dir + hidden)
                srnn_datas.append(srnn)
            img_dir = f'./image/compare/'
            draw_hidden_compare_img_version2(f'{img_dir}{fname}', run_num,
                                             nn_datas[1:-1],
                                             srnn_datas[:-1])
            #对隐层进行对比
            img_dir = f'./image/compare/'
            draw_hidden_compare_img_version2(f'{img_dir}{fname}',run_num,
                                             nn_datas[1:-1],
                                             srnn_datas[:-1])


            #读取对象计算值
            log_dir = f'log/{fname}_program.pkl'
            with open(log_dir, 'rb') as f:
                individual_Set = pickle.load(f)
            individual_Set = individual_Set[0]
            #重新画图
            range_func = _range_map[f'{fname}']
            kkk_func = _func_map[f'{fname}']
            arg = []
            y0 = np.random.uniform(2*range_func[0], 2*range_func[1], 2*range_func[2])[:, np.newaxis]
            #y0 = np.linspace(2*range_func[0], 2*range_func[1], 2*range_func[2])[:, np.newaxis]
            for num in range(kkk_func.arity):
                arg.append(y0)
            x_data = arg[0]
            flag = 0
            for a in arg:
                if flag > 0:
                    x_data = np.hstack((x_data,a))
                flag = flag + 1
            #true data
            t_data = kkk_func(*arg)
            y_datapre = x_data
            for hidden_num in range(n_layer-1):
                x_features = nn_datas[hidden_num].shape[1]
                y_features = nn_datas[hidden_num+1].shape[1]
                y_preSet = []
                y_datapre = predictResult(individual_Set[run_num][hidden_num], y_preSet, y_features, x_features,y_datapre)

            #network

            nn=load_nn_model(f'../ModelSave/{fname}.pth',nn=NN_MAP[f'{fname}']).cpu()
            x_data = torch.from_numpy(x_data).float()
            nn_predict = list(nn(x_data))[-1]
            nn_predict = nn_predict.detach().numpy()

            #画图
            plt.figure()
            plt.title(f'{fname}_{run_num}_out')
            plt.scatter(y0, y_datapre, label='sr',)
            plt.scatter(y0, t_data, label='true')
            plt.scatter(y0, nn_predict, label='nn')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc=10, bbox_to_anchor=(1.05, 1))
            plt.savefig(f'image/ts/{fname}_output_{run_num}.pdf', dpi=600, bbox_inches='tight')
            plt.show()
def image_trend_and_box(data_dir):
    #data_dir = 'D:/project/dataset2.0/'
    xlabel = 'K'

    refit_set = []
    for fname in os.listdir(data_dir):
        if fname[-1].isdigit() and fname[0] == 'k':
            with open(f'log/{fname}_program.pkl', 'rb') as f:
                records = pickle.load(f)
            records = records[0]
            refit = []
            for rec in records:
                #refit.append(rec[-1][0][0].raw_fitness_)
                refit.append(rec[-1][0][0].raw_fitness_)
            refit_set.append(refit)

            n_gen = 200
            with open(f'log/fitted/{fname}_30trend.json', 'r') as f:
                records = json.load(f)
            data_list = []
            for key, re in records.items():
                data_list.append(list(re))
            ylabel = 'fitness'
            draw_f_trend(f'image/{fname}_trend_test.pdf', n_gen, [data_list], legends=['srnn'], title=None, ylabel=ylabel)
    _draw_fitness_box(f'image/testbox_fit.pdf', refit_set, xlabel='K') #画出fitness_boximage

def getTrainResult(data_dir):

    for fname in os.listdir(data_dir):
        log_dir = f'log/{fname}_program.pkl'
        with open(log_dir, 'rb') as f:
            resultSet = pickle.load(f)
        print(resultSet)
if __name__ == '__main__':
    data_dir = '../Dataset/'
    #image_trend_and_box(data_dir) #画出fitness_trend 和 fitness_box
    #image_compare(data_dir) #比较隐层和输出层
    image_compareData3(data_dir)#比较隐层和输出层数据集3

    # getTrainResult(data_dir)
