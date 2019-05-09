import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取数据并归一化
def read_normalize(unit, o_path, d_path):
    path = o_path + unit
    o = open(path, 'rb')
    source_data = pd.read_csv(o, usecols=['open_count'], encoding='gbk')
    source_data = source_data.values

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler_series = scaler.fit_transform(source_data)
    return  source_data, scaler, scaler_series


# 构造数据集
# 通过之前的look_back个点来预callable后一个点
def create_dataset(dataset, look_back=10, test_size=0.3, random_state=0):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : i + look_back]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    x = np.array(dataX)
    y = np.array(dataY)
    x = x.astype(np.float32) # 很关键，默认为float64
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x, X_train, X_test, y_train, y_test


# 搭建模型
def create_model(X_train, X_test, y_train, look_back=10):
    X_train = X_train.reshape(-1, 1, look_back)
    y_train = y_train.reshape(-1, 1, 1)
    X_test = X_test.reshape(-1, 1, look_back)

    # Convert A NumPy Array To A PyTorch Tensor
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)

    # 定义模型
    class lstm_reg(nn.Module):                # 继承 torch 的 Module
        def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
            super(lstm_reg, self).__init__()  # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)   # LSTM
            self.reg = nn.Linear(hidden_size, output_size)            # 输出层线性输出

    #     一层层搭建(forward(x))层于层的关系链接
        def forward(self, x):
            x, _ = self.rnn(x)
            s, b, h = x.shape
            x = x.view(s*b, h)     # view: 返回一个有相同数据但大小不同的tensor
            x = self.reg(x)
            x = x.view(s, b, -1)
            return x

    net = lstm_reg(look_back, 1)
    return X_train, X_test, y_train, net


def fit_model(X_train, y_train, net):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    train_loss = []
    iteration = 100
    for i in tqdm(range(iteration)):
        var_x = Variable(X_train)
        var_y = Variable(y_train)

        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()     # 清空过往梯度
        loss.backward()           # 反向传播，计算当前梯度
        optimizer.step()          # 进行单次优化 (参数更新)
        train_loss.append(loss.item())   # 保存loss
        if (i + 1) % 100 == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(i + 1, loss.item()))


# 预测时间序列
def predict(net, look_back, x, scaler):
    net_pre = net.eval()
    xx = x.reshape(-1, 1, look_back)
    xx = torch.from_numpy(xx)
    var_data = Variable(xx)
    # 预测结果
    pred_test = net(var_data)
    pred_test = pred_test.reshape(1, -1)
    # 还原
    result_data = scaler.inverse_transform(pred_test.detach().numpy()).flatten()
    newIndex = range(look_back, len(result_data) + look_back)
    result = pd.Series(result_data, index=newIndex)
    return net_pre, result, result_data


# MAPE误差
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# # 预测是从第11个点开始的，所以这边的误差比较会忽略前10个点
# error = mean_squared_error(source_data[look_back:], result_data)
# print('Test MSE: %.3f' % error)
# error = mean_absolute_error(source_data[look_back:], result_data)
# print('Test MAE: %.3f' % error)
# error = mean_absolute_percentage_error(source_data[look_back:], result_data)
# print('Test MAPE: %.3f' % error)



# 找到指标最大的几个异常点，或指定一个指标，大于该指标的为异常点
def findAbnormalPoints(source, result, quota, MAE=False, MAPE=False):
    if MAE is True and MAPE is True:
        print("Parameter MAE and MAPE can not be True at the same time")
        return None
    if len(source) != len(result):
        print("The length of source and result must be the same")
        return None

    ab_index = []

    if MAE is True:
        for i in range(len(source)):
            if abs(source[i] - result[i]) > quota:
                ab_index.append(i)
    elif MAPE is True:
        for i in range(len(source)):
            if abs((source[i] - result[i]) / source[i]) > quota:
                ab_index.append(i)
    else:
        MAE_set = {}
        MAPE_set = {}
        MAE_min_key, MAPE_min_key = '-1', '-1'
        MAE_min_value, MAPE_min_value = -1, -1

        for i in range(len(source)):
            MAE_value = abs(source[i] - result[i])
            MAPE_value = abs((source[i] - result[i]) / source[i])

            if len(MAE_set) >= quota:
                if MAE_value > MAE_min_value:
                    MAE_set.pop(MAE_min_key)
                    MAE_set[str(i)] = MAE_value
                    MAE_min_key = min(MAE_set, key=MAE_set.get)
                    MAE_min_value = MAE_set[MAE_min_key]
            else:
                MAE_set[str(i)] = MAE_value
                # 看着很冗余，实际上可以减少运算次数
                MAE_min_key = min(MAE_set, key=MAE_set.get)
                MAE_min_value = MAE_set[MAE_min_key]

            if len(MAPE_set) >= quota:
                if MAPE_value > MAPE_min_value:
                    MAPE_set.pop(MAPE_min_key)
                    MAPE_set[str(i)] = MAPE_value
                    MAPE_min_key = min(MAPE_set, key=MAPE_set.get)
                    MAPE_min_value = MAPE_set[MAPE_min_key]
            else:
                MAPE_set[str(i)] = MAPE_value
                MAPE_min_key = min(MAPE_set, key=MAPE_set.get)
                MAPE_min_value = MAPE_set[MAPE_min_key]

        ab_index1 = []
        ab_index2 = []
        for key1, key2 in zip(MAE_set, MAPE_set):
            ab_index1.append(int(key1))
            ab_index2.append(int(key2))
        ab_index.append(ab_index1)
        ab_index.append(ab_index2)

    return ab_index

def show(ab_index, look_back, source_data, result_data):
    print(len(ab_index))
    if len(ab_index) == 2:
        for i in ab_index[0]:
            real_key = int(i) + look_back
            print('index = %d, true = %f, predicted = %f' % (real_key, source_data[real_key], result_data[int(i)]))
        print('\n')
        for i in ab_index[1]:
            real_key = int(i) + look_back
            print('index = %d, true = %f, predicted = %f' % (real_key, source_data[real_key], result_data[int(i)]))
    if len(ab_index) > 2:
        for i in ab_index:
            real_key = int(i) + look_back
            print('index = %d, true = %f, predicted = %f' % (real_key, source_data[real_key], result_data[int(i)]))


def lstm(unit, o_path, d_path, look_back):
    source_data, scaler, scaler_series = read_normalize(unit, o_path, d_path)
    x, X_train, X_test, y_train, y_test = create_dataset(scaler_series, look_back, 0.3, 0)
    X_train, X_test, y_train, net = create_model(X_train, X_test, y_train)
    fit_model(X_train, y_train, net)
    net_pre, result, result_data = predict(net, look_back, x, scaler)

    # 分别选取AE和APE最大的10个点
    ab_index = findAbnormalPoints(source_data[look_back:], result_data, 10)
    show(ab_index, look_back, source_data, result_data)

    real_index1 = [int(i) + look_back for i in ab_index[0]]
    real_index2 = [int(i) + look_back for i in ab_index[1]]

    ab_value1 = []
    for i in range(len(real_index1)):
        ab_value1.append(source_data[real_index1[i]])

    ab_value2 = []
    for i in range(len(ab_index[1])):
        ab_value2.append(source_data[real_index2[i]])

    plt.figure(figsize=(15,8))
    plt.plot(result, 'r', label='prediction')
    plt.plot(source_data, 'b', label='ture')
    plt.scatter(real_index1, ab_value1, s=200, marker='o', color='black', label='MAE_abnormal')
    plt.scatter(real_index2, ab_value2, s=200, marker='o', color='green', label='MAPE_abnormal')
    plt.legend(loc='best')
    plt.show()

    # 选取AE大于阈值的点
    ab_index = findAbnormalPoints(source_data[look_back:], result_data, 70, MAE=True)
    show(ab_index, look_back, source_data, result_data)

    real_index = [int(i) + look_back for i in ab_index]
    ab_value = []
    for i in real_index:
        ab_value.append(source_data[i])

    plt.figure(figsize=(15,8))
    plt.plot(result, 'r', label='prediction')
    plt.plot(source_data, 'b', label='ture')
    plt.scatter(real_index, ab_value, s=200, marker='o', color='black', label='abnormal')
    plt.legend(loc='best')
    plt.show()

    # 选取APE大于阈值的点
    ab_index = findAbnormalPoints(source_data[look_back:], result_data, 0.5, MAPE=True)
    show(ab_index, look_back, source_data, result_data)

    real_index = [int(i) + look_back for i in ab_index]
    ab_value = []
    for i in real_index:
        ab_value.append(source_data[i])

    plt.figure(figsize=(15,8))
    plt.plot(result, 'r', label='prediction')
    plt.plot(source_data, 'b', label='ture')
    plt.scatter(real_index, ab_value, s=200, marker='o', color='black', label='abnormal')
    plt.legend(loc='best')
    plt.show()
