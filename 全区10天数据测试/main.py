import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import datetime

# ----------------------------------------------- 1. 数据预处理  --------------------------------------------------
from preprocess import gather_data, split_by_neighbor, split_by_address, delete_redundancy, data_supplement

# 1. 读取所有原文件，并聚合为同一个文件
# column_names = ['device_id', 'address', 'neighbor', 'app_name', 'number', 'device_id1',
#                 'status', 'power', 'received_time', 'report_time', 'status_name', 'something']
# usecols = ['device_id','address', 'neighbor', 'app_name', 'status', 'received_time']
# o_path = '../dataset/origin/'
# d_path1 = '../dataset/gathered_units.csv'    # 住宅单元门
# d_path2 = '../dataset/gathered_doors.csv'    # 帮扶门室
# files = os.listdir(o_path)
# gathered_df1, gathered_df2 = None, None
# for file in files:
#     print(file)
#     df1 = gather_data(file, column_names, usecols, o_path, 0) # 最后为0表示住宅单元门
#     if gathered_df1 is None:
#         gathered_df1 = df1
#     else:
#         gathered_df1 = pd.concat([gathered_df1, df1], axis=0)
#
#     df2 = gather_data(file, column_names, usecols, o_path, 1) # 最后为1表示帮扶门室
#     if gathered_df2 is None:
#         gathered_df2 = df2
#     else:
#         gathered_df2 = pd.concat([gathered_df2, df2], axis=0)
# print(gathered_df1.info())
# print(gathered_df2.info())
# csvFile1 = open(d_path1, 'w')
# csvFile2 = open(d_path2, 'w')
# gathered_df1.to_csv(d_path1, index=None)
# gathered_df2.to_csv(d_path2, index=None)
# print('gathered')


# 2. 将聚合的数据其分小区和地址划分为csv文件
# o_path = '../dataset/gathered_units.csv'
# d_path = '../dataset/neighbors_units/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# split_by_neighbor( o_path, d_path )
# print('splited by neighbor')
#
# o_path = '../dataset/neighbors_units/'
# d_path = '../dataset/units/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# neighbors = os.listdir(o_path)
# for neighbor in neighbors:
#     split_by_address( neighbor, o_path, d_path )
# print('splited by address')


# 3. 删除重复数据：同一设备同一状态之间间隔再？s以内则删去重复者
# o_path = '../dataset/units/'
# d_path = '../dataset/redundancy_deleted/'
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         delete_redundancy(address, unit, o_path, d_path)
# print('deleted')

# 4. 补充数据：补充在一次超时事件中漏报的状态（正常开门，超市未关门报警，报警解除）
# o_path = '../dataset/redundancy_deleted/'
# d_path = '../dataset/complete_units/'
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         data_supplement(address, unit, o_path, d_path)
# print('supplement')

# ------------------------------------------------ 2. 状态计算  ---------------------------------------------------
# 1. 计算单元设备每小时/每日开门次数
# 2. 计算每个设备每个小时正常开门的总次数并转化为CSV文件（0时到1时之间的算作1时，以此类推）
# 3. 计算每个设备每个天正常开门的总次数并转化为CSV文件
# 4. 将逐小时数据按天分开成单独的csv文件，方便绘图
# 5. 将逐天数据按周分开成单独的csv文件，方便绘图
# 6. 计算单元设备总开门次数(工作日工作时段，工作日夜间，周末日间，凌晨)
# 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长

from counters import rooms_open_counter, delete_little_open, open_frequency_peer_hour,open_frequency_peer_day,\
    split_by_day, split_by_week,total_open_condition,open_to_close_time, spring_festival_open

# # 1. 计算单元设备每日开门次数
# o_path = '../dataset/complete_units/'
# d_path = '../counts/1day_origin/'
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         rooms_open_counter(address, unit, o_path, d_path)
# print('rooms_open_counted')

# 这个功能还没有试！！！！！！！！！！！！！！！！！！！！！！！
# 将超过5天开门次数为0的设备剔除保存
# o_path = '../counts/1day_origin/'
# d_path = '../counts/1day/'
# d_path_deleted = '../counts/1day_deleted/'
# df, i = pd.DataFrame(columns=['neighbor', 'address', 'zero_count']), 0
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         zero_count = delete_little_open(address, unit, o_path, d_path_deleted)
#         if zero_count > 5:
#             df.loc[i] = {'neighbor':unit, 'address':address.split('.')[0], 'zero_count':zero_count}
#             i += 1
#             shutil.copy(o_path + unit + '/' + address, d_path+ unit + '/' + address)
# print('delete_little_open')
#
# # 2. 计算单元设备每小时开门次数，并将超过5天开门次数为0的设备剔除保存#
# o_path = '../dataset/complete_units/'
# d_path_day = '../counts/1day/'
# d_path_hour = '../counts/1hour/'
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         rooms_open_counter(address, unit, o_path, d_path_day)
#     for address in addresses:
#         rooms_open_counter(address, unit, o_path, d_path_hour)
# print('rooms_open_counted')

# 3. 计算每个设备每个小时正常开门的总次数并转化为CSV文件（0时到1时之间的算作1时，以此类推）
# o_path = '../counts/1hour/'
# d_path = '../counts/peer_hour/'
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         open_frequency_peer_hour(address, unit, o_path, d_path)
# print('open_frequency_peer_hour')
#
# # 4. 计算一周中每天正常开门的总次数并转化为CSV文件
# o_path = '../counts/1day/'
# d_path = '../counts/peer_day/'
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         open_frequency_peer_day(address, unit, o_path, d_path)
# print('open_frequency_peer_day')
#
#
# # 4. 将逐小时数据按天分开成单独的csv文件，方便绘图
# o_path = '../counts/1hour/'
# d_path = '../counts/days_split/'
# if not os.path.exists(d_path):
# 	os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
# 	split_by_day(unit, o_path, d_path)
# print('split_by_day')
#
# # 5. 将逐天数据按周分开成单独的csv文件，方便绘图
# o_path = '../counts/1day/'
# d_path = '../counts/weeks_split/'
# if not os.path.exists(d_path):
# 	os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
# 	split_by_week(unit, o_path, d_path)
# print('split_by_day')
#
# # 6. 计算单元设备总开门次数(工作日工作时段，工作日夜间，周末日间，凌晨)
# start_time = datetime.datetime.now()
# o_path = '../dataset/complete_units/'
# d_path_total = '../counts/total_open_condition.csv'
# d_path_unit =  '../counts/total_open_conditions/'
# if not os.path.exists(d_path_unit):
#     os.makedirs(d_path_unit)
# df_total = pd.DataFrame(columns = ['neighbor', 'workday_workhours', 'workday_commuting_hours', 'weekend_daytime', 'wee_hours'])
# i = 0   # 控制df_total的索引
# units = os.listdir(o_path)
# for unit in units:
#     df_unit = pd.DataFrame(columns = ['neighbor', 'address', 'workday_workhours', 'workday_commuting_hours', 'weekend_daytime', 'wee_hours'])
#     j = 0   # 控制df_unit的索引
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         total_open_condition(address, unit, o_path, df_unit, j)
#         j += 1
#     csvFile1 = open(d_path_unit + unit + '.csv', 'w')
#     df_unit.to_csv(d_path_unit + unit + '.csv', index=None)
#     df_total.loc[i] = {'neighbor': df_unit.loc[0]['neighbor'], 'workday_workhours': df_unit['workday_workhours'].sum(), 'workday_commuting_hours': df_unit['workday_commuting_hours'].sum(),
#                        'weekend_daytime': df_unit['weekend_daytime'].sum(), 'wee_hours': df_unit['wee_hours'].sum()}
#     i += 1
# csvFile2 = open(d_path_total, 'w')
# df_total.to_csv(d_path_total, index=None, encoding='utf-8')
# print('total_open_condition')
# end_time = datetime.datetime.now()
# print('start_time: ', start_time, 'end_time: ', end_time, 'during: ',end_time - start_time)

# # 7. 提取出春节期间开门次数数据
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
# print('spring_festival_open')

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

# 画出春节期间每天开门次数折线图
# o_path= '../counts/spring_festival/1day/'
# d_path = '../imgs/spring_festival/day/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     plt_spring_days(unit, o_path, d_path)

# ---------------------------------  4. 根据历史数据均值方差画出最佳曲线和上下界曲线  --------------------------------
from statistical import model

# o_path_weekly = '../counts/peer_day/'
# o_path_daily = '../counts/peer_hour/'
# d_path_weekly = '../statistical_model/weekly/'
# d_path_workday = '../statistical_model/daily_workday/'
# d_path_weekend = '../statistical_model/daily_weekend/'
# # units_weekly = os.listdir(o_path_weekly)
# # for unit in units_weekly:
# #     addresses = os.listdir(o_path_weekly + unit + '/')
# #     for address in addresses:
# #         model(address, unit, o_path_weekly, d_path_weekly)
# units_daily = os.listdir(o_path_daily)
# for unit in units_daily:
#     addresses = os.listdir(o_path_daily + unit + '/')
#     for address in addresses:
#         model(address, unit, o_path_daily, d_path_workday)
#     for address in addresses:
#         model(address, unit, o_path_daily, d_path_weekend)

# ------------------------------------------------ 5. 聚类分析  ----------------------------------------------------
# 画出层次聚类树状图，输出聚类结果，画出每一类的设备表现图
from cluster import cluster
# from cluster1 import cluster1

o_path = '../counts/total_open_condition.csv' 	#开门不同时段
d_path = '../clusters/'
if not os.path.exists(d_path):
    os.makedirs(d_path)
cluster(o_path, d_path)

# ------------------------------------------------ 6. 异常检测  ---------------------------------------------------
from LSTM import lstm, hourly_anomaly_corr_rate

# 使用lstm模型进行预测并检测异常
# o_path = '../counts/1hour/'
# d_path = '../anomaly_result/hourly/anomalies/'
# look_back = 5
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(o_path + unit + '/')
#     for address in addresses:
#         lstm(address, unit, o_path_weekly, d_path_weekly)

# 使用【均值±3*标准差】检验异常检测的正确率
# o_path = '../anomaly_result/hourly/anomalies/'
# corr_anomalies_path = '../anomaly_result/hourly/corr_anomalies/'
# workday_path = '../statistical_model/daily_workday/'
# weekend_path = '../statistical_model/daily_weekend/'
# df_corr_rate = pd.DataFrame(columns=['address', 'corr_rate'])
# i = 0
#
# units = os.listdir(o_path)
# for unit in units:
#     addresses = os.listdir(unit)
#     for address in addresses:
#         corr_rate = hourly_anomaly_corr_rate(o_path + unit, workday_path, weekend_path ,corr_anomalies_path)
#         address_name = address.split('.')[0]
#         df_corr_rate.loc[i] = {'neighbor':unit, 'address': address, 'corr_rate': corr_rate}
#         i += 1
# csvFile = open('../anomaly_result/hourly/corr_rate.csv', 'w')
# df_corr_rate.to_csv('../anomaly_result/hourly/corr_rate.csv')
# mean_corr_rate = df_corr_rate['corr_rate'].mean()
# print('时刻异常检测平均正确率: ', mean_corr_rate)

