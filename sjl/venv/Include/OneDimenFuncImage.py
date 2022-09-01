
import numpy as np
import matplotlib.pyplot as plt

def drawOneDimensional(all_data,all_y_dataPre,all_nn,alltrue,y_dataPreFinal,i,funName,allFitness):
    '''
    画图
    :param all_data:数据x轴
    :param all_y_dataPre: 符号回归预测值
    :param all_nn: 神经网络预测值
    :param alltrue: 真实值
    :param y_dataPreFinal: 直接符号回归预测值
    :param i: 运行到次数
    :param funName:函数名
    :return:
    '''
    fig= plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    bx = fig.add_subplot(2, 2, 2)
    cx = fig.add_subplot(2, 2, 3)
    dx = fig.add_subplot(2, 2, 4)
    title = "fitness is "
    title = title+str(allFitness)
    ax.set(title=title)
    ax.plot(all_data, all_y_dataPre, 'b-', lw=3, label='sr')
    bx.plot(all_data, all_nn, 'g-', lw=3, label='network neutral')
    cx.plot(all_data, alltrue, 'y-', lw=3, label='true')
    dx.plot(all_data, y_dataPreFinal, 'm-', lw=3, label='sr-direct')
    # plt.plot(fint, otest, 'g-', lw=3,label='network neutral')
    ax.legend()
    bx.legend()
    cx.legend()
    dx.legend()
    imageName = 'log/'+str(funName)+'/image/single'+str(i+1)+'.png'
    plt.savefig(imageName,  dpi=600,format='png')
    #plt.show()
    plt.close(fig)
    # 总图
    fig =plt.figure()
    fig.suptitle(title)
    plt.plot(all_data, all_y_dataPre, 'b-', lw=1, label='sr')
    plt.plot(all_data, alltrue, 'y-', lw=1, label='true')
    plt.plot(all_data, all_nn, 'g-', lw=1, label='network neutral')
    plt.plot(all_data, y_dataPreFinal, 'm-', lw=1, label='sr-direct')
    plt.legend()
    imageName = 'log/'+str(funName)+'/image/whole' + str(i + 1) + '.png'
    plt.savefig(imageName,  dpi=600,format='png')
    plt.close(fig)