import numpy as np

def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大，正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    '''if several > 0:

        several_min_or_max = np_arr[np.argpartition(np_arr,several)[:several]]
    else:'''

    several_min_or_max = np_arr[np.argpartition(np_arr,1)[:1]]
    several_min_or_ma = np_arr[np.argpartition(np_arr,-1)[-1:]]
    return several_min_or_max, several_min_or_ma

if __name__ == "__main__":
    np_arr = np.array([11, 7, 19,9, 3, 5, 8, 8, 0, 3, 2,1,22,20])
    print(get_Several_MinMax_Array(np_arr, 6))

