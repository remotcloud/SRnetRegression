from sklearn import datasets,linear_model,discriminant_analysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#加载数据
def load_data():
    diabetes=datasets.load_diabetes()
    return train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

#定义线性回归模型
def test_LinearRegrssion(*data):
    x_train,x_test,y_train,y_test=data
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)#训练数据
    print('Coefficients:' ,regr.coef_[1])#权重向量即为每个特征的相关系数
    print("Residual sum of square:%.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))#均方误差，每个特征的（预测值-真实值的平方）的平均值
    print('Score:%.2f' % regr.score(x_test, y_test))#得分
x_train,x_test,y_train,y_test=load_data()
test_LinearRegrssion(x_train,x_test,y_train,y_test)
