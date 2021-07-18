# -*- coding=utf-8 -*-
# @Time :2021/6/16 11:11
# @Author :Hobbey
# @Site :
# @File : LR.py
# @Software : PyCharm


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
        theta_x = np.exp(x) / (1 + np.exp(x))
        return theta_x


class My_LogisticRegression(object):
    def __init__(self, penalty='l2', tol=0.0001, C=1.0, bais=True, max_iter=10000, learning_rate=0.01):
        '''
        :param penalty: 'l1' or 'l2' 默认：'l2' 选择正则化方式
        :param tol: min_error 默认：1e-4 迭代停止的最小的误差
        :param C: 默认：1.0 惩罚系数
        :param bias: 默认：True 是否需要偏差 b
        :param max_iter: 默认：10000最大迭代次数(学习次数)
        :param learning_rate 默认：0.01 学习率
        '''
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.bais = bais
        self.max_iter = max_iter
        self.learning_rate=learning_rate
        # 待训练的参数
        self.w_trained=None
        

    # 总控函数
    def fit(self, X_train, y_train):
        '''
        :param x_train: 训练数据集
        :param y_train: 训练标签集 对应数据集的标签 简单应当是二分类的 0，1
        :return: this module
        '''
        X_train, y_train = self.datapreprocess(X_train, y_train) # 数据预处理
        self.fit_transform(X_train, y_train) # 调用训练函数fit
        


    # 数据预处理
    def datapreprocess(self, X, y):
        # 转为numpy
        X_train = np.array(X)
        y_train = np.array(y)
        # 数据标准化(归一化)处理
        scale = StandardScaler()#标准化数据通过减去均值然后除以方差
        X_train = scale.fit_transform(X_train)

        return X_train, y_train
        
    
    # 训练函数
    def fit_transform(self, X_train, y_train):
        '''
        :param X_train: 训练数据集
        :param y_train: 训练标签集 对应数据集的标签 简单应当是二分类的 0，1
        :return: this module
        '''
        assert len(X_train)==len(y_train),'数据集和标签集数量不一致'
        assert len(label_set:=np.unique(y_train))==2,'标签集不是二分类的' # 生成警告

        # 初始化偏置项与权重矩阵
        np.random.seed(0)
        if self.bais == True:
            self.w_ini = np.random.randn(X_train.shape[1] + 1) # 随机初始化：30特征项对应权重参数k+bias偏置项
            X = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1) # 添加偏置项
        else:
            self.w_ini = np.random.randn(X_train.shape[1]) # 随机初始化：30特征项对应权重参数k
            X = X_train

        # 随机梯度下降
        w_t = self.w_ini
        for i in range(self.max_iter):
            # 调用函数cost_function，计算损失
            diff = self.cost_function(X, y_train, w_t)

            # 退出方式：如果精度超过0.9就退出
            # 使用列表生成式简化代码
            y_prob_t = [sigmoid(np.dot(w_t, x)) for x in X]  # 生成S函数概率矩阵
            y_pred_t = [1 if y > 0.5 else 0 for y in y_prob_t]  # 概率>0.5，判为1；否则为0
            mistake_index = np.where(y_pred_t != y_train)[0] # 分类错的索引下标
            mistake_num = len(mistake_index) # 原先矩阵w_t分类错误的个数
            if 1 - mistake_num / len(y_pred_t) > 0.9:
                break

            # 调用函数gradient_function，根据cost_function计算得到的梯度，梯度下降更新权重矩阵w
            w_t = self.gradient_function(w_t, diff)

        self.w_trained = w_t # 训练参数绑定给对象
        
        
    # 预测函数
    def predict(self, X_test, y_test):
        '''
        :param X_test: 测试集
        :return: y_pred: 预测标签
        '''
        # 数据预处理
        X_test, y_test = self.datapreprocess(X_test, y_test) 
        # 预测
        if self.bais == True:
            X = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1) # 添加偏置项
        else:
            X = X_test
        y_prob = [sigmoid(np.dot(self.w_trained, x)) for x in X] # 生成S函数概率矩阵
        y_pred = [1 if y > 0.5 else 0 for y in y_prob] # 概率>0.5，判为1；否则为0
        self.pred =  y_pred # 赋给对象本身
        
        # 返回预测结果
        return self.pred 

        
    # 精度计算函数
    def score(self, X_test, y_test):
        '''
        :param X_test: 测试集
        :param y_test: 测试标签
        :return: 正确率
        '''
        # 数据预处理
        X_test, y_test = self.datapreprocess(X_test, y_test) 
        # 预测
        if self.bais == True:
            X = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1) # 添加偏置项
        else:
            X = X_test
        y_prob = [sigmoid(np.dot(self.w_trained, x)) for x in X] # 生成S函数概率矩阵
        y_pred = [1 if y > 0.5 else 0 for y in y_prob] # 概率>0.5，判为1；否则为0
        self.pred =  y_pred # 赋给对象本身
        # 对预测的结果打分
        mistake_index = np.where(self.pred != y_test)[0] # 分类错的索引下标
        mistake_num = len(mistake_index) # 分类错误的个数

        return 1 - mistake_num / len(self.pred) # 正确率
        
        
    # 损失函数
    def cost_function(self, X, y, w_t):
        # 1.交叉熵误差
        j = (np.random.randint(0, X.shape[0])) # 随机在训练集中选取数据
        diff = sigmoid(-y[j] * np.dot(w_t, X[j])) * (-y[j] * X[j])  


        return diff
        

    # 系数更新函数
    def step(self, X, y):
        """
        填补此处
        """
        pass
        

    # 梯度下降函数
    def gradient_function(self, w_t, diff):
        '''
        :param w_t: 旧权重矩阵
        :param diff: 当前旧矩阵计算得到的损失(又称梯度)
        :return:
        '''
        w_t1 = w_t - self.learning_rate * diff # 计算新的权重矩阵
        w_t = w_t1 # 更新权重矩阵参数

        return w_t



path = r'D:\\Summercamp\\week1\\LogisticRegression\\'


if __name__ == '__main__':
    # data = np.loadtxt(path + r'breast_cancer.csv', dtype=np.float64, delimiter=',')
    # x_train = data[:, :-1]
    # y_train = data[:, -1]

    # 使用pandas读取数据，不将第一行作为特征列
    data = pd.read_csv(path + r'breast_cancer.csv', header=None)
    X_train = data.iloc[:, :-1] # 训练集特征, Dataframe格式
    y_train = data.iloc[:, -1] # 训练集标签, Series格式

    # 建立对象，调用总控函数
    # 我没有写带惩罚系数的两种正则化，你可以自己写一下
    # 由于调试时我发现，两次之间的w相差太小，所以我也没采取w_t1-w_t < tol就退出的方式，而是采用精度>0.9就退出训练的方式
    LR = My_LogisticRegression(penalty='l2', bais=True, max_iter=100000, learning_rate=0.01) 
    LR.fit(X_train=X_train.copy(), y_train=y_train.copy())
    print('自己写的Logisitic回归精度为:')
    print(LR.score(X_train, y_train)) # 没有测试集就只能拿训练集测试了

    # y_pred = LR.predict(X_test, y_test)

    # 对比scikit-learn库自带的逻辑回归
    from sklearn.linear_model import LogisticRegression
    LR2 = LogisticRegression(penalty='l2', C=0.5, solver='liblinear')
    LR2.fit(X_train, y_train)
    print('scikit-learn的Logisitic回归精度为:')
    print(LR2.score(X_train, y_train))