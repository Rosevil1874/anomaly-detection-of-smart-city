import numpy as np
import pandas as pd
import os
from datetime import timedelta
from dateparsers import dateparse1, dateparse2, dateparse3, dateparse4
from collections import Counter


# 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def open_to_close_time(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=["received_time"], date_parser=dateparse2)
	df_out = pd.DataFrame(columns = ['start_time', 'end_time', 'duration'])
	i, idx = 0, 0			# 控制原df的遍历，控制新df_out的新行添加

	# 遍历查找事件的开始时间（超时的前五分钟-正常开门）和结束时间（报警解除）
	while idx < len(df):
		start_time, end_time = None, None
		if df.loc[idx]['status'] == 6:
			start_time = df.loc[idx]['received_time'] - timedelta(minutes = 5)
			idx += 1
			for j in range(idx, len(df)):
				if df.loc[j]['status'] == 0:
					continue
				elif df.loc[j]['status'] == 7:
					end_time = df.loc[j]['received_time']
					idx = j + 1
					break
				else:
					break
			else:
				idx += 1
		else:
			idx += 1

		# 计算事件的时长
		if start_time and end_time:
			duration = round ((end_time - start_time).seconds/3600, 2)		# 将time_delta转化为秒再转化为小时为单位（保留两位小数）
			df_out.loc[i] = {'start_time': start_time, 'end_time': end_time, 'duration': duration}
			i += 1

	new_path = d_path + unit
	csvFile = open(new_path, 'w')
	df_out.to_csv(new_path, index=None)


# 计算设备超时事件在每个小时开始的次数和时长
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def hourly_open_to_close_time(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=["start_time"], date_parser=dateparse2)

	hours = [i for i in range(24)]
	counts = [0]*24
	durations = [[] for i in range(24)]
	min_duration, max_duration = [0]*24, [0]*24

	for idx in df.index:
		start_time = df.loc[idx]['start_time']
		hour_idx = start_time.hour
		counts[ hour_idx ] += 1
		durations[ hour_idx ].append(df.loc[idx]['duration'])
		if len(durations[ hour_idx ]) != 0:
			min_duration[ hour_idx ] = min(durations[ hour_idx ])
			max_duration[ hour_idx ] = max(durations[ hour_idx ])

	# for i in range(len(durations)):
		# print(durations[i])
	result = {}
	result['hours'] = hours
	result['counts'] = counts
	result['durations'] = durations
	result['min_duration'] = min_duration
	result['max_duration'] = max_duration
	result_df = pd.DataFrame(result)
	temp = result_df.pop('hours')
	result_df.insert(0, 'hours', temp)
	result_df.to_csv(d_path + unit, index=None)


# 根据超时事件在每个小时开始的次数和时长为每个设备计算报警规则
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def unit_alarm_rules(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o)
	rules = [0]*24

	for idx in df.index:
		if df.loc[idx]['counts'] == 0:
			rules[idx] = 1
		else:
			durations = df.loc[idx]['durations'].strip('[]')
			durations = [float(x)for x in durations.split(',')]
			round_durations = [np.round(x) for x in durations]
			frequency = dict(Counter(round_durations))
			frequent_durations = []		# 保存大于1小时的出现三次及以上的时长
			for k,v in frequency.items():
				if k> 1 and v >= 3:
					frequent_durations.append(k)
			if len(frequent_durations) != 0:
				rules[idx] = max(frequent_durations)
			else:
				rules[idx] = max(int(np.mean(durations)), 1)

	rules_df = {}
	rules_df['hours'] = [i for i in range(24)]
	rules_df['rules'] = rules
	rules_df = pd.DataFrame(rules_df)
	rules_df.to_csv(d_path + unit, index=None)


# 计算根据新的规则报警率下降情况
# 【
# 	unit：单元设备地址；
#  	before_path: 原超时报警数据路径；
#  	after_path：新报警规则路径；
#  】
# return: 每个设备运用新规则后报警降低率
def alarm_reduce_rate(unit, before_path, after_path):
	before = before_path + unit
	after = after_path + unit
	o1 = open(before, 'rb')
	o2 = open(after, 'rb')
	df_before = pd.read_csv(o1)
	df_after = pd.read_csv(o2)
	count_before = df_before['counts'].sum()
	rules = df_after['rules'].as_matrix()
	# print(count_before, rules[0])
	count_after = 0
	for idx in df_before.index:
		if df_before.loc[idx]['counts'] > 0:
			durations = df_before.loc[idx]['durations'].strip('[]')
			durations = [float(x) for x in durations.split(',')]
			count_after += sum(i > rules[idx] for i in durations)
	# reduce_rate = format((count_before - count_after) / count_before, '.2%')
	reduce_rate = (count_before - count_after) / count_before
	return reduce_rate
