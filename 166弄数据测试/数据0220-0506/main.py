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


# --------------------------------------------- 7. 根据超时时长设置报警规则  ---------------------------------------------------
# 1. 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长；
# 2. 计算设备超时事件在每个小时发生的次数和时长；
# 3. 根据异常检测结果提取出表现正常的每小时数据;计算设备超时事件在每个小时内发生的次数和时长，以及最大、最小、平均时长
# 4. 根据平均超时时长设置新的报警规则；
# 5. 计算根据新的规则报警率下降情况。


from alram_v3 import open_to_close_time,  hourly_open_to_close_time, \
    timeSlot_unit_alarm_rules,timeSlot_open_to_close_time
from alram_reduce_rate import timeSlot_alarm_reduce_rate
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


# 3. 根据异常检测结果提取出表现正常的每小时数据;
# 计算设备超时事件在每个小时内发生的次数和时长，以及最大、最小、平均时长
# timeout_path = '../counts/open_to_close_time/'
# d_path = '../alarm_rules/hourly_open_to_close_time/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(timeout_path)
# for unit in units:
#     hourly_open_to_close_time(unit, timeout_path, d_path)
#
#
# # 将小时数据结合成分时段数据
# o_path = '../alarm_rules/hourly_open_to_close_time/'
# d_path = '../alarm_rules/timeSlot_open_to_close_time/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     timeSlot_open_to_close_time(unit, o_path, d_path)


# 4. 根据平均超时时长设置新的报警规则；
# o_path = '../alarm_rules/timeSlot_open_to_close_time/'
# d_path = '../alarm_rules/timeSlot_unit_alarm_rules/'
# if not os.path.exists(d_path):
#     os.makedirs(d_path)
# units = os.listdir(o_path)
# for unit in units:
#     timeSlot_unit_alarm_rules(unit, o_path, d_path)


# 5. 合并规则
# 合并规则：报警规则类似的设备使用同一套规则 TODO：优化合并规则代码
# o_path = '../alarm_rules/timeSlot_unit_alarm_rules/'
# d_path = '../alarm_rules/timeSlot_total_unit_alarm_rules.csv'
# cluster_path = '../alarm_rules/timeSlot_auto_clusters/'
# merged_path = '../alarm_rules/timeSlot_auto_merged/'

# 将所有设备规则放入同一文件
# units = os.listdir(o_path)
# total_rules = pd.DataFrame(columns=['address', '0-5', '5-7', '7-10', '10-11', '11-13', '13-17', '17-20', '20-24'])
# i = 0
# for unit in units:
#     path = o_path + unit
#     o = open(path, 'rb')
#     df = pd.read_csv(o)
#     new_line = np.append(np.array([unit.split('.')[0]]), df['rules'].as_matrix())
#     total_rules.loc[i] = new_line
#     i += 1
# csvFile = open(d_path, 'w')
# total_rules.to_csv(d_path, index=None)
#
# # 先将没有发生超时的设备分（告警规则全为5分钟）离出来，剩下的进行聚类
# timeout_alarm_rules = pd.DataFrame(columns=['address', '0-5', '5-7', '7-10', '10-11', '11-13', '13-17', '17-20', '20-24'])
# non_timeout_alarm_rules = pd.DataFrame(columns=['address', '0-5', '5-7', '7-10', '10-11', '11-13', '13-17', '17-20', '20-24'])
# total_rules = pd.read_csv(d_path, encoding='gbk')
# i, j = 0, 0
# for idx in total_rules.index:
#     rule = total_rules.loc[idx]
#     if list(rule[1:].as_matrix()) == [5, 5, 5, 5, 5, 5, 5, 5]:
#         non_timeout_alarm_rules.loc[i] = rule
#         i += 1
#     else:
#         timeout_alarm_rules.loc[j] = rule
#         j += 1
# csvFile1 = open('../alarm_rules/timeout_alarm_rules.csv', 'w')
# csvFile2 = open('../alarm_rules/non_timeout_alarm_rules.csv', 'w')
# timeout_alarm_rules.to_csv(csvFile1, index=None)
# non_timeout_alarm_rules.to_csv(csvFile2, index=None)

# DBSCAN聚类
# 手肘法调参
# timeout_alarm_rules = pd.read_csv('../alarm_rules/timeout_alarm_rules.csv', index_col='address', encoding='gbk')
# eps_list = list(range(50, 1000, 10))
# SC = []
# for i in eps_list:
#     db = DBSCAN(eps=i, min_samples=1).fit(timeout_alarm_rules)
#     labels = db.labels_
#     n_labels = len(set(labels))
#     if n_labels == 1:
#         break
#     else:
#         SC.append(metrics.silhouette_score(timeout_alarm_rules, labels))
# best_eps = eps_list[np.argmax(SC)]
# # print(SC)
# # print(np.argmax(SC))
# # print(best_eps)
# # db = DBSCAN(eps=best_eps, min_samples=1).fit(total_rules)
# db = DBSCAN(eps=best_eps, min_samples=1).fit(timeout_alarm_rules)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# k = len(set(labels))									# 聚类簇的数量
# # print(labels)
# #详细输出原始数据及其类别
# if not os.path.exists(cluster_path + 'imgs/'):
#     os.makedirs(cluster_path + 'imgs/')
# if not os.path.exists(cluster_path + 'csv/'):
#     os.makedirs(cluster_path + 'csv/')
#
# r = pd.concat([timeout_alarm_rules, pd.Series(labels, index = timeout_alarm_rules.index)], axis = 1)  #详细输出每个样本对应的类别
# r.columns = list(timeout_alarm_rules.columns) + [u'聚类类别']              #重命名表头
# plt.rcParams['font.sans-serif'] = ['SimHei']                #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False                  #用来正常显示负号
# # style = ['ro-', 'go-', 'bo-', 'co-', 'mo-', 'yo-']
# xlabels = ['0-5', '5-7', '7-10', '10-11', '11-13', '13-17', '17-20', '20-24']
# pic_output = cluster_path + 'imgs/type_'                 	#聚类图文件名前缀
# for i in range(k):  #逐一作图，作出不同样式
#     plt.figure()
#     tmp = r[r[u'聚类类别'] == i].iloc[:,:8]                    #提取每一类除最后一列（label）的数据
#     tmp.to_csv( cluster_path + 'csv/类别%s.csv' %(i) )     		#将每一类存成一个csv文件
#     for j in range(len(tmp)):                                 #作图
#         # plt.plot( range(1, 5), tmp.iloc[j], style[i - 1] )
#         plt.plot( range(1, 9), tmp.iloc[j] )
#         plt.xticks( range(1, 9), xlabels, rotation = 20 )         #坐标标签
#         plt.title( u'门洞类别%s' % (i) )                         	#从1开始计数
#         plt.subplots_adjust( bottom=0.15 )                        #调整底部
#         plt.savefig( u'%s%s.png' % (pic_output, i) )            	#保存图片


# 将每个类别中的规则合并成一个规则：对应每个小时取众数，若每个值出现次数一样则取中位数
# classes_path = cluster_path + 'csv/'
# classes = os.listdir(classes_path)
# merged_cols = ['class', '0-5', '5-7', '7-10', '10-11', '11-13', '13-17', '17-20', '20-24' ]
# merged_df = pd.DataFrame(columns=merged_cols)
# i = 0
# for c in classes:
#     path = classes_path + c
#     o = open(path, 'rb')
#     df = pd.read_csv(o, encoding='gbk')
#     new_line = []
#     for col in merged_cols:
#         if col == 'class':
#             new_line.append(c.split('.')[0])
#         else:
#             rules = df[col].as_matrix()
#             new_line.append(np.median(rules))  # 中位数
#             # counts = np.bincount(rules)
#             # mode = np.argmax(counts)        # 众数
#             # if np.max(counts) == 1:
#             #     new_line.append(np.median(rules))   # 中位数
#             # else:
#             #     new_line.append(mode)
#     merged_df.loc[i] = new_line
#     i += 1
# if not os.path.exists(merged_path):
#     os.makedirs(merged_path)
# csvFile = open(merged_path + 'merged_rules.csv', 'w')
# # print(merged_df)
# merged_df.to_csv(csvFile, index=None)


# 每个类别中包含的设备整理成一个文件
# o_path = '../alarm_rules/timeSlot_auto_clusters/csv/'
# d_path = '../alarm_rules/timeSlot_auto_merged/classes_to_devices.csv'
# non_timeout_path = '../alarm_rules/non_timeout_alarm_rules.csv'   # 未发生超时的设备
# classes_list, devices_list = [], []
# clusters = os.listdir(o_path)
# for cluster in clusters:
#     o = open(o_path + cluster, 'rb')
#     df = pd.read_csv(o, encoding='gbk')
#     classes_list.append(cluster.split('.')[0])
#     devices_list.append(df['address'].as_matrix())
#
# o = open(non_timeout_path, 'rb')
# non_timeout_df = pd.read_csv(o, encoding='gbk')
# non_timeout_devices = non_timeout_df['address'].as_matrix()
# devices_list.append(non_timeout_devices)
# classes_list.append('无超时')
# classes_dict = {}
# classes_dict['class'] = classes_list
# classes_dict['devices'] = devices_list
# classes_dict = pd.DataFrame(classes_dict)
# csvFile = open(d_path, 'w')
# classes_dict.to_csv(d_path, index=None)


# 画出合并后规则图像
# auto_merged_rules_path = '../alarm_rules/timeSlot_auto_merged/merged_rules.csv'
# o = open(auto_merged_rules_path, 'rb')
# auto_merged_rules_df = pd.read_csv(o, index_col='class', encoding='gbk')

# 总图
# fig2, ax2 = plt.subplots(figsize=(16,9))
# lines2 = []
# for c in auto_merged_rules_df.index:
#     print(c)
#     line, = ax2.plot(auto_merged_rules_df.columns, auto_merged_rules_df.loc[c], label=c)
#     lines2.append(line)
# ax2.set_xlabel('时刻')
# ax2.set_ylabel('超时时长上限')
# plt.legend(lines2, auto_merged_rules_df.index)
# plt.show()
# plt.savefig('../alarm_rules/timeSlot_auto_merged/merged_rules.png')
# plt.close(fig2)

# 每类别独立图
# auto_img_path = '../alarm_rules/timeSlot_auto_merged/auto_imgs/'
# if not os.path.exists(auto_img_path):
#     os.makedirs(auto_img_path)
# for c in auto_merged_rules_df.index:
#     fig, ax = plt.subplots()
#     line, = ax.plot(auto_merged_rules_df.columns, auto_merged_rules_df.loc[c], label=c)
#     # ax.axhline(y=60, color='g', linestyle='--')  # 添加y = 60的参考线
#     ax.set_xlabel('时刻')
#     ax.set_ylabel('超时时长上限')
#     plt.suptitle(c)
#     # plt.ylim(0, 70)
#     plt.savefig( auto_img_path + c + '.png')
#     plt.close(fig)


# 5. 计算根据新的规则报警率下降情况 (按时段计算超时情况的下降率计算)
before_path = '../counts/open_to_close_time/'
rules_path = '../alarm_rules/timeSlot_auto_merged/merged_rules.csv'
classes_path = '../alarm_rules/timeSlot_auto_merged/classes_to_devices.csv'
d_path = '../alarm_rules/auto_alarm_reduce_rate.csv'
o1, o2 = open(rules_path, 'rb'), open(classes_path, 'rb')
df_rules, df_classes = pd.read_csv(o1, index_col='class', encoding='gbk'), pd.read_csv(o2, encoding='gbk')
df_result, i = pd.DataFrame(columns = ['address', 'reduce_rate']), 0

units = os.listdir(before_path)
for unit in units:
    reduce_rate = timeSlot_alarm_reduce_rate(unit, before_path, df_rules, df_classes)
    if reduce_rate != 0:
        df_result.loc[i] = {'address': unit.split('.')[0], 'reduce_rate': reduce_rate}
        i += 1
csvFile = open(d_path, 'w')
df_result.to_csv(d_path, index=None)
print('mean reduce rate:', df_result['reduce_rate'].mean())



# 5. 计算根据新的规则报警率下降情况 (原按小时计算超时情况的下降率计算)
# before_path = '../counts/open_to_close_time/'
# rules_path = '../alarm_rules/auto_merged/merged_rules.csv'
# classes_path = '../alarm_rules/auto_merged/classes_to_devices.csv'
# d_path = '../alarm_rules/auto_alarm_reduce_rate.csv'
# o1, o2 = open(rules_path, 'rb'), open(classes_path, 'rb')
# df_rules, df_classes = pd.read_csv(o1, index_col='class', encoding='gbk'), pd.read_csv(o2, encoding='gbk')
# df_result, i = pd.DataFrame(columns = ['address', 'reduce_rate']), 0
#
# # alarm_reduce_rate('岚皋路166弄2号.csv', before_path, df_rules, df_classes)
#
# units = os.listdir(before_path)
# for unit in units:
#     reduce_rate = alarm_reduce_rate(unit, before_path, df_rules, df_classes)
#     if reduce_rate != 0:
#         df_result.loc[i] = {'address': unit.split('.')[0], 'reduce_rate': reduce_rate}
#         i += 1
# csvFile = open(d_path, 'w')
# df_result.to_csv(d_path, index=None)
# print('mean reduce rate:', df_result['reduce_rate'].mean())
