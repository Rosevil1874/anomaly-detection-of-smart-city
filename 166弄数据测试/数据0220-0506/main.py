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
# o_path = '../dataset/门磁.xlsx'
# d_path = '../dataset/units/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# usecols = ['address', 'app_name', 'status', 'received_time']
# split_by_address(usecols, o_path, d_path)


# 3. 删除重复数据：同一设备同一状态之间间隔再15s以内则删去重复者
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
    split_by_day, split_by_week,total_open_condition, spring_festival_open


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

# ---------------------------------  4. 根据历史数据均值方差画出最佳曲线和上下界曲线  --------------------------------
from statistical import model

# ------------------------------------------------ 5. 聚类分析  ----------------------------------------------------

# ------------------------------------------------ 6. 异常检测 LSTM  ---------------------------------------------------
from counters import del_much_zeros
from LSTM_class import lstm_class, hourly_anomaly_corr_rate

# ------------------------------------------------ 6. 异常检测 FFT  ---------------------------------------------------
from FFT import anomaly_detection,daily_anomaly_corr_rate, hourly_anomaly_corr_rate, gather_anomalies


# --------------------------------------------- 7. 根据超时时长设置报警规则  ---------------------------------------------------
# 1. 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长；
# 2. 计算设备超时事件在每个小时发生的次数和时长；
# 3. 根据异常检测结果提取出表现正常的每小时数据;计算设备超时事件在每个小时内发生的次数和时长，以及最大、最小、平均时长
# 4. 根据平均超时时长设置新的报警规则；
# 5. 计算根据新的规则报警率下降情况。

from alram_v2 import open_to_close_time, hourly_start_open_to_close_time, hourly_open_to_close_time, \
    unit_alarm_rules, alarm_reduce_rate
from scipy.cluster.hierarchy import distance,linkage,dendrogram,fcluster
from sklearn.cluster import DBSCAN
from sklearn import metrics


# 1. 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长
# o_path = '../dataset/complete_units/'
# d_path = '../counts/open_to_close_time/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     open_to_close_time(unit, o_path, d_path)


# 5. 计算根据新的规则报警率下降情况 TODO
before_path = '../counts/open_to_close_time/'
rules_path = '../alarm_rules/auto_merged/merged_rules.csv'
classes_path = '../alarm_rules/auto_merged/classes_to_devices.csv'
d_path = '../alarm_rules/auto_alarm_reduce_rate.csv'
o1, o2 = open(rules_path, 'rb'), open(classes_path, 'rb')
df_rules, df_classes = pd.read_csv(o1, index_col='class', encoding='gbk'), pd.read_csv(o2, encoding='gbk')
df_result, i = pd.DataFrame(columns = ['address', 'reduce_rate']), 0

# alarm_reduce_rate('岚皋路166弄2号.csv', before_path, df_rules, df_classes)

units = os.listdir(before_path)
for unit in units:
    reduce_rate = alarm_reduce_rate(unit, before_path, df_rules, df_classes)
    if reduce_rate != 0:
        df_result.loc[i] = {'address': unit.split('.')[0], 'reduce_rate': reduce_rate}
        i += 1
csvFile = open(d_path, 'w')
df_result.to_csv(d_path, index=None)
print('mean reduce rate:', df_result['reduce_rate'].mean())
