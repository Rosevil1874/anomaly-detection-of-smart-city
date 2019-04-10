import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

# ----------------------------------------------- 1. 数据预处理  --------------------------------------------------
from preprocess import gather_data, split_by_address, delete_redundancy, data_supplement

# 1. 读取所有原文件，并聚合为同一个文件
# column_names = ['device_id', 'address', 'neighbor', 'app_name', 'number', 'device_id1',
#                 'status', 'power', 'received_time', 'report_time', 'status_name', 'something']
# usecols = ['address', 'app_name', 'status', 'received_time']
# o_path = '../dataset/origin/'
# d_path = '../dataset/gathered.csv'
# files = os.listdir(o_path)
# gathered_df = None
# for file in files:
#     print(file)
#     df = gather_data(file, column_names, usecols, o_path)
#     if gathered_df is None:
#         gathered_df = df
#     else:
#         gathered_df = pd.concat([gathered_df, df], axis=0)
# print(gathered_df.info())
# csvFile = open(d_path, 'w')
# gathered_df.to_csv(d_path, index=None)


# 2. 将聚合的数据其按地址划分为csv文件
# o_path = '../dataset/door0301.xlsx'
# d_path = '../dataset/units/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# usecols = ['address', 'app_name', 'status', 'received_time']
# split_by_address(usecols, o_path, d_path)


# 3. 删除重复数据：同一设备同一状态之间间隔再？s以内则删去重复者
# o_path = '../dataset/units/'
# d_path = '../dataset/redundancy_deleted/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     delete_redundancy(unit, o_path, d_path)

# 4. 补充数据：补充在一次超时事件中漏报的状态（正常开门，超市未关门报警，报警解除）
# o_path = '../dataset/redundancy_deleted/'
# d_path = '../dataset/complete_units/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     data_supplement(unit, o_path, d_path)

# ------------------------------------------------ 2. 状态计算  ---------------------------------------------------
# 1. 计算单元设备每小时/每日开门次数
# 2. 计算每个设备每个小时正常开门的总次数并转化为CSV文件（0时到1时之间的算作1时，以此类推）
# 3. 计算每个设备每个天正常开门的总次数并转化为CSV文件
# 4. 将逐小时数据按天分开成单独的csv文件，方便绘图
# 5. 将逐天数据按周分开成单独的csv文件，方便绘图
# 6. 计算单元设备总开门次数(工作日工作时段，工作日夜间，周末日间，凌晨)

from counters import rooms_open_counter, open_frequency_peer_hour,open_frequency_peer_day,\
    split_by_day, split_by_week,total_open_condition,open_to_close_time, spring_festival_open

# 1. 计算单元设备每小时/每日开门次数
# o_path = '../dataset/complete_units/'
# d_path_day = '../counts/1day/'
# d_path_hour = '../counts/1hour/'
# if not os.path.exists(d_path_day):
#     os.makedirs(d_path_day)
# if not os.path.exists(d_path_hour):
#     os.makedirs(d_path_hour)
# #
# units = os.listdir(o_path)
# for unit in units:
#     rooms_open_counter(unit, o_path, d_path_day)
# for unit in units:
#     rooms_open_counter(unit, o_path, d_path_hour)


# 2. 计算每个设备每个小时正常开门的总次数并转化为CSV文件（0时到1时之间的算作1时，以此类推）
# o_path = '../counts/1hour/'
# d_path = '../counts/peer_hour/'
# units = os.listdir(o_path)
# if not os.path.exists(d_path):
# 	os.makedirs( d_path )
# for unit in units:
# 	open_frequency_peer_hour(unit, o_path, d_path)


# 3. 计算一周中每天正常开门的总次数并转化为CSV文件
# o_path = '../counts/1day/'
# d_path = '../counts/peer_day/'
# units = os.listdir(o_path)
# if not os.path.exists(d_path):
# 	os.makedirs( d_path )
# for unit in units:
# 	open_frequency_peer_day(unit, o_path, d_path)


# 4. 将逐小时数据按天分开成单独的csv文件，方便绘图
# o_path = '../counts/1hour/'
# d_path = '../counts/days_split/'
# if not os.path.exists(d_path):
# 	os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
# 	split_by_day(unit, o_path, d_path)

# 5. 将逐天数据按周分开成单独的csv文件，方便绘图
# o_path = '../counts/1day/'
# d_path = '../counts/weeks_split/'
# if not os.path.exists(d_path):
# 	os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
# 	split_by_week(unit, o_path, d_path)

# 6. 计算单元设备总开门次数(工作日工作时段，工作日夜间，周末日间，凌晨)
# o_path = '../dataset/complete_units/'
# d_path = '../counts/total_open_condition.csv'
# units = os.listdir(o_path)
# df_out = pd.DataFrame(columns = ["address", 'workday_workhours', 'workday_commuting_hours', 'weekend_daytime', 'wee_hours'])
# i = 0
# for unit in units:
# 	total_open_condition(unit, df_out, i, o_path)
# 	i += 1
# csvfile = open(d_path, 'w')
# df_out.to_csv(d_path, index=None, encoding='utf-8')

# 7. 提取出春节期间开门次数数据
# o_path_day = '../counts/1day/'
# d_path_day = '../counts/spring_festival/1day/'
# o_path_hour = '../counts/1hour/'
# d_path_hour = '../counts/spring_festival/1hour/'
# if not os.path.exists(d_path_day):
#     os.makedirs(d_path_day)
# if not os.path.exists(d_path_hour):
#     os.makedirs(d_path_hour)
# units = os.listdir(o_path_hour)
# for unit in units:
#     spring_festival_open(unit, o_path_day, d_path_day)
# units = os.listdir(o_path_hour)
# for unit in units:
#     spring_festival_open(unit, o_path_hour, d_path_hour)


# ----------------------------------------------- 3. 画图分析  ----------------------------------------------------
# 画出每个设备一天中每个小时对应的开门次数
# 画出每个设备一周中每天对应的开门次数
# 画出每个设备工作日/周末的一天开门次数模型
# 画出每个设备一周开门次数模型
# 画出每个设备一次超时事件开门时长(1小时及以内，1~3小时，3~6小时，6~10，10小时以上)占比
# 计算所有设备一次超时事件开门时长占比并画出饼图
# 画出春节期间每天开门次数折线图

from plt_analysis import day_img, print_day_img, week_img, print_week_img, \
    workday_peer_hour, week_peer_day, plt_open_to_close_time, plt_total_open_to_close_time, \
    plt_spring_days

# o_path= '../counts/spring_festival/1day/'
# d_path = '../imgs/spring_festival/day/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     plt_spring_days(unit, o_path, d_path)

# ---------------------------------  4. 根据历史数据均值方差画出最佳曲线和上下界曲线  --------------------------------
from statistical import model

# o_path_daily = '../counts/peer_hour/'
# o_path_weekly = '../counts/peer_day/'
# d_path_workday = '../statistical_model/daily_workday/'
# d_path_weekend = '../statistical_model/daily_weekend/'
# d_path_weekly = '../statistical_model/weekly/'
# units_daily = os.listdir(o_path_daily)
# for unit in units_daily:
# 	model(unit, o_path_daily, d_path_workday)
# for unit in units_daily:
# 	model(unit, o_path_daily, d_path_weekend)
#
# units_weekly = os.listdir(o_path_weekly)
# for unit in units_daily:
# 	model(unit, o_path_weekly, d_path_weekly)

# ------------------------------------------------ 5. 聚类分析  ----------------------------------------------------
# 画出层次聚类树状图，输出聚类结果，画出每一类的设备表现图
# from cluster import cluster
# # from cluster1 import cluster1
#
# o_path = '../counts/total_open_condition.csv' 	#开门不同时段
# d_path = '../clusters/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# cluster(o_path, d_path)

# ------------------------------------------------ 6. 异常检测 LSTM  ---------------------------------------------------
from counters import del_much_zeros
from LSTM_class import lstm_class, hourly_anomaly_corr_rate


# 将四分之三以上数据为0的设备剔除
# o_path = '../counts/1hour/'
# d_path1 = '../counts/1hour_much_zeros/'
# d_path2 = '../counts/1hour_little_zeros/'
# if not os.path.exists(d_path1):
#     os.makedirs(d_path1)
# if not os.path.exists(d_path2):
#     os.makedirs(d_path2)
# units = os.listdir(o_path)
# for unit in units:
#     del_much_zeros(unit, o_path, d_path1, d_path2)


start_time = datetime.datetime.now()
# 训练LSTM模型并根据MAE和MAPE检测异常
# o_path = '../counts/1hour_little_zeros/'
# d_path = '../anomaly_result/hourly/LSTM/anomalies/'
# img_path = '../anomaly_result/hourly/LSTM/imgs/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# if not os.path.exists(img_path):
#     os.makedirs(img_path)
# units = os.listdir(o_path)
# look_back = 5
# abnormal_rate = 0.03    # 找出数据表现最异常的3%
# for unit in units:
#     lstm_class(unit, o_path, d_path, img_path, abnormal_rate)
# end_time = datetime.datetime.now()
# print('start_time: ', start_time, 'end_time: ', end_time, 'during: ',end_time - start_time)

# 根据均值标准差模型检验异常检测正确率
# corr_anomalies_path = '../anomaly_result/hourly/LSTM/corr_anomalies/'
# d_path = '../anomaly_result/hourly/LSTM/anomalies/'
# if not os.path.exists(corr_anomalies_path):
#     os.makedirs(corr_anomalies_path)
# workday_path = '../statistical_model/daily_workday/'
# weekend_path = '../statistical_model/daily_weekend/'
# df_corr_rate = pd.DataFrame(columns=['address', 'corr_rate'])
# i = 0
# anomaly_devices = os.listdir(d_path)
# for device in anomaly_devices:
#     corr_rate = hourly_anomaly_corr_rate(d_path + device, workday_path, weekend_path ,corr_anomalies_path)
#     address = device.split('.')[0]
#     df_corr_rate.loc[i] = {'address': address, 'corr_rate': corr_rate}
#     i += 1
# csvFile = open('../anomaly_result/hourly/LSTM/corr_rate.csv', 'w')
# df_corr_rate.to_csv('../anomaly_result/hourly/LSTM/corr_rate.csv')
# mean_corr_rate = df_corr_rate['corr_rate'].mean()
# print('时刻异常检测平均正确率: ', mean_corr_rate)
# end_time = datetime.datetime.now()
# print('start_time: ', start_time, 'end_time: ', end_time, 'during: ',end_time - start_time)
#
# # 将异常结果汇总成一个表
# o_path = '../anomaly_result/hourly/LSTM/corr_anomalies/'
# d_path = '../anomaly_result/hourly/LSTM/total.csv'
# hourly_df = None
# hourly = os.listdir(o_path)
# for unit in hourly:
#     o = open(o_path + unit, 'rb')
#     df = pd.read_csv(o, encoding='gbk')
#     if hourly_df is None:
#         hourly_df = df
#     else:
#         hourly_df = pd.concat([hourly_df, df], axis=0)
# daily = os.listdir(o_path)
# hourlyFile = open(d_path, 'w')
# del hourly_df['Unnamed: 0']
# hourly_df.to_csv(d_path, index = None)
# end_time = datetime.datetime.now()
# print('start_time: ', start_time, 'end_time: ', end_time, 'during: ',end_time - start_time)

# ------------------------------------------------ 6. 异常检测 FFT  ---------------------------------------------------
from FFT import anomaly_detection,daily_anomaly_corr_rate, hourly_anomaly_corr_rate, gather_anomalies

# # daily异常检测
# o_path = '../counts/1day/'
# d_path = '../anomaly_result/daily/anomalies/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     anomaly_detection( unit, o_path, d_path)
#
# # 异常检测正确率
# corr_anomalies_path = '../anomaly_result/daily/corr_anomalies/'
# if not os.path.exists(corr_anomalies_path):
#     os.makedirs(corr_anomalies_path)
# weekly_path = '../statistical_model/weekly/'
# df_corr_rate = pd.DataFrame(columns=['address', 'corr_rate'])
# i = 0
# anomaly_devices = os.listdir(d_path)
# for device in anomaly_devices:
#     corr_rate = daily_anomaly_corr_rate(d_path + device, weekly_path, corr_anomalies_path)
#     address = device.split('.')[0]
#     df_corr_rate.loc[i] = {'address': address, 'corr_rate': corr_rate}
#     i += 1
# csvfile = open('../anomaly_result/daily/corr_rate.csv', 'w')
# df_corr_rate.to_csv('../anomaly_result/daily/corr_rate.csv')
# mean_corr_rate = df_corr_rate['corr_rate'].mean()
# print('日期异常检测平均正确率: ', mean_corr_rate)
#
# # hourly异常检测
# o_path = '../counts/1hour/'
# d_path = '../anomaly_result/hourly/anomalies/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     anomaly_detection(unit, o_path, d_path)
#
# corr_anomalies_path = '../anomaly_result/hourly/corr_anomalies/'
# if not os.path.exists(corr_anomalies_path):
#     os.makedirs(corr_anomalies_path)
# workday_path = '../statistical_model/daily_workday/'
# weekend_path = '../statistical_model/daily_weekend/'
# df_corr_rate = pd.DataFrame(columns=['address', 'corr_rate'])
# i = 0
# anomaly_devices = os.listdir(d_path)
# for device in anomaly_devices:
#     corr_rate = hourly_anomaly_corr_rate(d_path + device, workday_path, weekend_path ,corr_anomalies_path)
#     address = device.split('.')[0]
#     df_corr_rate.loc[i] = {'address': address, 'corr_rate': corr_rate}
#     i += 1
# csvFile = open('../anomaly_result/hourly/corr_rate.csv', 'w')
# df_corr_rate.to_csv('../anomaly_result/hourly/corr_rate.csv')
# mean_corr_rate = df_corr_rate['corr_rate'].mean()
# print('时刻异常检测平均正确率: ', mean_corr_rate)

# 将异常结果汇总成一个表，并加上地址、周几字段
# hourly_o_path = '../anomaly_result/hourly/corr_anomalies/'
# daily_o_path = '../anomaly_result/daily/corr_anomalies/'
# hourly_d_path = '../anomaly_result/hourly/total.csv'
# daily_d_path = '../anomaly_result/daily/total.csv'
# hourly_df = None
# daily_df = None
# hourly = os.listdir(hourly_o_path)
# for unit in hourly:
#     df = gather_anomalies(unit, hourly_o_path)
#     if hourly_df is None:
#         hourly_df = df
#     else:
#         hourly_df = pd.concat([hourly_df, df], axis=0)
# daily = os.listdir(daily_o_path)
# for unit in daily:
#     df = gather_anomalies(unit, daily_o_path)
#     if daily_df is None:
#         daily_df = df
#     else:
#         daily_df = pd.concat([daily_df, df], axis=0)
# hourlyFile = open(hourly_d_path, 'w')
# dailyFile = open(daily_d_path, 'w')
# hourly_df.to_csv(hourly_d_path, index = None)
# daily_df.to_csv(daily_d_path, index = None)


# --------------------------------------------- 7. 根据超时时长设置报警规则  ---------------------------------------------------
# 1. 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长

from counters import open_to_close_time

o_path = '../dataset/complete_units/'
d_path = '../counts/open_to_close_time/'
if not os.path.exists(d_path):
    os.makedirs(d_path)
units = os.listdir(o_path)
for unit in units:
    open_to_close_time(unit, o_path, d_path)
