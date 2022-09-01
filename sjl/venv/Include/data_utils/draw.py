import matplotlib.pyplot as plt
import numpy as np


def draw_range_line(x,
                    ys_mins,
                    ys_maxs,
                    ys_means,
                    xlabel,
                    ylabel,
                    legends,
                    savefile=None,
                    ):

    plt.figure()

    for y_mins, y_maxs, y_means, l in zip(ys_mins, ys_maxs, ys_means, legends):
        plt.plot(x, y_means, label=l)
        plt.fill_between(x, y_mins, y_maxs, alpha=0.4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=2, markerscale=20, bbox_to_anchor=(1.05, 1))
    plt.savefig(savefile, dpi=600, bbox_inches='tight')
    plt.show()


def draw_2d_graph_and_save(x, y, savepath=None, title='training loss graph', xlabel='epoch', ylabel='loss'):
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()


def draw_mul_graph_and_save(x, y1, y2,
                            savepath=None,
                            title='test and prediction graph',
                            label=['prediction', 'test label']):

    plt.title(title)
    plt.scatter(x.view(-1), y1.view(-1), label=label[0])
    plt.scatter(x.view(-1), y2.view(-1), label=label[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()


def draw_submul_graph_and_save(x_samples, y_samples1, y_samples2, subcol, savepath, title='test and prediction graph', label=['prediction', 'test label']):
    fig, axs = plt.subplots(1, subcol, sharey=True)
    fig.suptitle(title)

    for j in range(subcol):
        axs[j].plot(x_samples[j], y_samples1[j], label=label[0])
        axs[j].plot(x_samples[j], y_samples2[j], label=label[1])
    plt.legend(loc='best')
    plt.savefig(savepath)
    plt.show()


def project_to_2d_and_save(vars: tuple, zs: tuple, savefile,
                           vars_labels: list,
                           y_label,
                           zs_legends: list):
    if vars_labels is None:
        vars_labels = [f'x{i}' for i in range(len(vars))]
    if y_label is None:
        y_label = 'y'
    fig = plt.figure()
    n_var, n_z = len(vars), len(zs)
    if n_var == 2:
        # plot 3d
        n_row, begin, end = 2, 2, 4
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        for z, legend in zip(zs, zs_legends):
            ax3d.scatter(*vars, z, label=legend, s=0.1)
        ax3d.set_xlabel(vars_labels[0])
        ax3d.set_ylabel(vars_labels[1])
        ax3d.set_zlabel(y_label)
    else:
        n_row, begin, end = n_var // 2 + 1 if n_var % 2 > 0 else n_var // 2, 1, n_var+1

    var_idx = 0
    for i in range(begin, end):
        ax2d = fig.add_subplot(n_row, 2, i)
        for z, legend in zip(zs, zs_legends):
            ax2d.scatter(vars[var_idx], z, label=legend, s=0.1)
        ax2d.set_xlabel(vars_labels[var_idx])
        ax2d.set_ylabel(y_label)
        var_idx += 1

    plt.legend(loc=2, markerscale=20, bbox_to_anchor=(1.05, 1))
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()

