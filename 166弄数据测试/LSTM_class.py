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
from dateparsers import dateparse2, dateparse4

class LSTM:
    def __init__(self, source_data, look_back, fit_iteration):
        self.look_back = look_back          # 用前look_back个历史数据预测第look_back+1个数据
        self.source_data = source_data      # 转化成ndarray的源数据
        self.scaler = MinMaxScaler(feature_range=(0,1))     # 归一化标准
        self.scaler_series = []             # 归一化后的数据
        self.X_train = []                   # 样本特征训练集
        self.X_test = []                    # 样本特征测试集
        self.y_train = []                   # 样本结果训练集
        self.y_test = []                    # 样本结果测试集
        self.net = None                     # 用来保存LSTM网络
        self.fit_iteration = fit_iteration  # 训练数据时迭代的次数
        self.result_data = []               # 预测结果
        self.ab_index = []                  # 异常检测结果（异常的索引）


    # 构造数据集
    def create_dataset(self, test_size = 0.3, random_state=0):
        # 归一化
        self.scaler_series = self.scaler.fit_transform(self.source_data)
        dataX, dataY = [], []
        for i in range(len(self.scaler_series) - self.look_back):
            a = self.scaler_series[i : i + self.look_back]
            dataX.append(a)
            dataY.append(self.scaler_series[i + self.look_back])
        x = np.array(dataX)
        y = np.array(dataY)
        x = x.astype(np.float32) # 很关键，默认为float64
        y = y.astype(np.float32)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x

    # 构造模型
    def create_model(self):
        self.X_train = self.X_train.reshape(-1, 1, self.look_back)
        self.y_train = self.y_train.reshape(-1, 1, 1)
        self.X_test = self.X_test.reshape(-1, 1, self.look_back)

        # Convert A NumPy Array To A PyTorch Tensor
        self.X_train = torch.from_numpy(self.X_train)
        self.y_train = torch.from_numpy(self.y_train)
        self.X_test = torch.from_numpy(self.X_test)

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

        self.net = lstm_reg(self.look_back, 1)


    # 训练模型
    def fit_model(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.01)
        train_loss = []
        for i in tqdm(range(self.fit_iteration)):
            var_x = Variable(self.X_train)
            var_y = Variable(self.y_train)

            # 前向传播
            out = self.net(var_x)
            loss = criterion(out, var_y)
            # 反向传播
            optimizer.zero_grad()     # 清空过往梯度
            loss.backward()           # 反向传播，计算当前梯度
            optimizer.step()          # 进行单次优化 (参数更新)
            train_loss.append(loss.item())   # 保存loss
            if (i + 1) % 100 == 0:
                print('Epoch: {}, Loss: {:.5f}'.format(i + 1, loss.item()))

    # 预测
    def predict(self, x):
        net_pre = self.net.eval()
        xx = x.reshape(-1, 1, self.look_back)
        xx = torch.from_numpy(xx)
        var_data = Variable(xx)
        # 预测结果
        pred_test = self.net(var_data)
        pred_test = pred_test.reshape(1, -1)
        # 还原
        self.result_data = self.scaler.inverse_transform(pred_test.detach().numpy()).flatten()
        newIndex = range(self.look_back, len(self.result_data) + self.look_back)
        result = pd.Series(self.result_data, index=newIndex)
        return net_pre, result


    # 找到指标最大的几个异常点，或指定一个指标，大于该指标的为异常点
    def findAbnormalPoints(self, source, quota, MAE=False, MSE=False):
        if MAE is True and MSE is True:
            print("Parameter MAE and MSE can not be True at the same time")
            return None
        if len(source) != len(self.result_data):
            print("The length of source and result must be the same")
            return None

        ab_index = []
        # 找到大于阈值quota的索引
        if MAE is True:
            for i in range(len(source)):
                if abs(source[i] - self.result_data[i]) > quota:
                    ab_index.append(i)
        elif MSE is True:
            for i in range(len(source)):
                if abs((source[i] - self.result_data[i]) / source[i]) > quota:
                    ab_index.append(i)

        # 找到值最大的quota个索引
        else:
            MAE_set = {}
            MSE_set = {}
            MAE_min_key, MSE_min_key = '-1', '-1'
            MAE_min_value, MSE_min_value = -1, -1

            for i in range(len(source)):
                MAE_value =  mean_absolute_error([source[i]], [self.result_data[i]])
                MSE_value = mean_squared_error([source[i]], [self.result_data[i]])

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

                if len(MSE_set) >= quota:
                    if MSE_value > MSE_min_value:
                        MSE_set.pop(MSE_min_key)
                        MSE_set[str(i)] = MSE_value
                        MSE_min_key = min(MSE_set, key=MSE_set.get)
                        MSE_min_value = MSE_set[MSE_min_key]
                else:
                    MSE_set[str(i)] = MSE_value
                    MSE_min_key = min(MSE_set, key=MSE_set.get)
                    MSE_min_value = MSE_set[MSE_min_key]

            ab_index1 = []
            ab_index2 = []
            for key1, key2 in zip(MAE_set, MSE_set):
                ab_index1.append(int(key1))
                ab_index2.append(int(key2))
            # ab_index.append(ab_index1)
            # ab_index.append(ab_index2)
            # 求两个list的并集
            ab_index = list(set(ab_index1).union(set(ab_index2)))
        self.ab_index = ab_index


    # 显示异常检测结果
    def show(self, result, unit, img_path):
        ab_index = self.ab_index
        look_back = self.look_back
        source_data = self.source_data
        result_data = self.result_data

        print( '异常个数：', len(ab_index))
        ab_value = []
        real_index = [int(i) + look_back for i in ab_index]
        for i in real_index:
            ab_value.append(source_data[i])
            print('index = %d, true = %f, predicted = %f' % (i, source_data[i], result_data[i - look_back]))
        plt.figure(figsize=(15,8))
        plt.plot(result, 'r', label='prediction')
        plt.plot(source_data, 'b', label='ture')
        plt.scatter(real_index, ab_value, s=200, marker='o', color='black', label='abnormal')
        plt.legend(loc='best')
        address = unit.split('.')[0]
        plt.suptitle(address)
        plt.savefig(img_path + address + '.png')
        plt.close()
        # plt.show()

# 读取数据
def read_data(unit, o_path):
    path = o_path + unit
    o1 = open(path, 'rb')
    o2 = open(path, 'rb')
    source_df = pd.read_csv(o1, parse_dates=['received_time'], date_parser=dateparse2, encoding='gbk')
    source_data = pd.read_csv(o2, usecols=['open_count'], encoding='gbk')
    # source_data = source_data.values
    # print(source_data['open_count'].values)
    return source_df, source_data.values

# 保存结果
def save_result(source_data, ab_index, look_back, d_path):
    real_index = [int(i) + look_back for i in ab_index]
    abnormal_df, abnormal_idx = pd.DataFrame(columns=['address', 'received_time', 'weekday', 'day_type', 'period_of_time', 'open_count']), 0
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    address = d_path.split('/')[-1].split('.')[0]
    for idx in real_index:
        item = source_data.loc[idx]
        received_time = item['received_time']
        weekday = weekdays[received_time.weekday()]
        open_count = item['open_count']

        # 判断日期属性，节假日包括（元旦，五一，国庆，圣诞）
        month, day, day_type = received_time.month, received_time.day, ''
        if month == 1 and day == 1 or month == 5 and day >= 1 and day <= 3 or month == 10 and day >= 1 and day <= 7 or month == 12 and day == 25:
            day_type = '节假日'
        elif weekday == 5 or weekday == 6:
            day_type = '周末'
        else:
            day_type = '工作日'

        # 判断时间段
        hour, period_of_time = received_time.hour, ''
        if hour >=0 and hour <= 4:
            period_of_time = '凌晨'
        elif hour <= 8:
            period_of_time = '早上'
        elif hour <= 10:
            period_of_time = '早通勤'
        elif hour <= 17:
            period_of_time = '工作时段'
        elif hour <= 19:
            period_of_time = '晚通勤'
        else:
            period_of_time = '晚上'
        abnormal_df.loc[abnormal_idx] = {'address': address, 'received_time': received_time, 'weekday':weekday, 'day_type':day_type, 'period_of_time':period_of_time, 'open_count':open_count}
        abnormal_idx += 1
    csvFile = open(d_path, 'w' )
    abnormal_df.to_csv(d_path, index=None)

# 检测正确性
def hourly_anomaly_corr_rate(device_o_path, workday_o_path, weekend_o_path, corr_anomalies_o_path):
    address = device_o_path.split('/')[-1]
    o = open(workday_o_path + address.split('.')[0] + '/' + address)
    statistical_workday = pd.read_csv(o, index_col=['received_time'])
    o = open(weekend_o_path + address.split('.')[0] + '/' + address)
    statistical_weekend = pd.read_csv(o, index_col=['received_time'])
    o = open(device_o_path)
    anomaly = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2)

    corr_count, i = 0, 0
    cols = anomaly.columns.values.tolist()
    df_corr = pd.DataFrame(columns = cols)
    df_corr['anomaly_type'] = ''
    for idx in anomaly.index:
        week = anomaly.loc[idx]['received_time'].weekday()
        hour = anomaly.loc[idx]['received_time'].hour
        open_count = anomaly.loc[idx]['open_count']
        #     周末
        if week ==5 or week == 6:
            u_bound = statistical_weekend.loc[str(hour + 1) + '时']['u_bound']
            l_bound = statistical_weekend.loc[str(hour + 1) + '时']['l_bound']
            if open_count > u_bound :
                corr_count += 1
                df_corr.loc[i] = anomaly.loc[idx]
                df_corr.loc[i,'anomaly_type'] = '过高'
                i += 1
            elif open_count < l_bound:
                corr_count += 1
                df_corr.loc[i] = anomaly.loc[idx]
                df_corr.loc[i,'anomaly_type'] = '过低'
                i += 1

        #     工作日
        else:
            u_bound = statistical_workday.loc[str(hour + 1) + '时']['u_bound']
            l_bound = statistical_workday.loc[str(hour + 1) + '时']['l_bound']
            if open_count > u_bound :
                corr_count += 1
                df_corr.loc[i] = anomaly.loc[idx]
                df_corr.loc[i,'anomaly_type'] = '过高'
                i += 1
            elif open_count < l_bound:
                corr_count += 1
                df_corr.loc[i] = anomaly.loc[idx]
                df_corr.loc[i,'anomaly_type'] = '过低'
                i += 1

    csvFile = open(corr_anomalies_o_path + address, 'w')
    df_corr.to_csv(corr_anomalies_o_path + address)
    corr_rate = corr_count / len(anomaly)
    return corr_rate


# 使用LSTM模型进行异常检测
def lstm_class(unit, o_path, d_path, img_path, abnormal_rate):
    source_df, source_data = read_data(unit, o_path)
    lstm_model = LSTM(source_data, 10, 500)
    x = lstm_model.create_dataset(test_size=0.3, random_state=0)
    lstm_model.create_model()
    lstm_model.fit_model()
    net_pre, result = lstm_model.predict(x)
    lstm_model.findAbnormalPoints(lstm_model.source_data[lstm_model.look_back:], len(source_df)*abnormal_rate)
    lstm_model.show(result, unit, img_path)
    ab_index, look_back = lstm_model.ab_index, lstm_model.look_back
    save_result(source_df, ab_index, look_back, d_path + unit)
