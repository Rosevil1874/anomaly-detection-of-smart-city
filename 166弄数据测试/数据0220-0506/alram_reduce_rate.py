import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
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

		# 计算事件的时长
		if start_time and end_time:
			duration = round ((end_time - start_time).total_seconds()/3600, 2)		# 将time_delta转化为秒再转化为小时为单位（保留两位小数）
			df_out.loc[i] = {'start_time': start_time, 'end_time': end_time, 'duration': duration}
			i += 1

	new_path = d_path + unit
	csvFile = open(new_path, 'w')
	df_out.to_csv(new_path, index=None)


# 计算设备超时事件在每个小时发生的次数和时长
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def hourly_start_open_to_close_time(unit, o_path, d_path):
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


# 根据异常检测结果提取出表现正常的每小时数据
# 【
# 	unit：单元设备地址；
#  	anomalies_path: 异常数据路径；
#  	timeout_path：超时数据路径；
#  	days_path：按天划分的数据集，用来获取该设备共有多少天的数据；
#  	d_path：结果存储的路径；
#  】
def hourly_open_to_close_time(unit, anomalies_path, timeout_path, days_path, d_path):
	path1, path2, path3 = anomalies_path + unit, timeout_path + unit, days_path + unit
	o1, o2, o3 = open(path1, 'rb'), open(path2, 'rb'), open(path3, 'rb')
	anomalies_df = pd.read_csv(o1, parse_dates=['received_time'], date_parser=dateparse2, encoding='gbk')
	timeout_df = pd.read_csv(o2, parse_dates=['start_time', 'end_time'], date_parser=dateparse2)
	days_df = pd.read_csv(o3)
	days = len(days_df)  # 该设备共有days天的数据

	abnormal_hours = anomalies_df['received_time']
	abnormal_hours = [x.strftime('%Y-%m-%d %H') for x in abnormal_hours]

	hours = [i for i in range(24)]
	durations = [0] * 24
	counts = [0] * 24
	min_duration = [float('inf')] * 24
	max_duration = [float('-inf')] * 24

	for idx in timeout_df.index:
		start_time = timeout_df.loc[idx]['start_time']
		end_time = timeout_df.loc[idx]['end_time']
		delta = end_time - start_time
		delta = int(np.ceil(delta.total_seconds() / 60 / 60))  # timedelta化成小时为单位
		if delta > 24: continue;		# 忽略超过24小时的超时事件

		for i in range(delta):
			# 若是开门表现异常时间，则忽略当前小时的数据
			if start_time.strftime('%Y-%m-%d %H') in abnormal_hours:
				continue;
			if i == 0:
				if start_time.hour == end_time.hour:
					diff = end_time.minute - start_time.minute
					durations[start_time.hour] += diff
					min_duration[start_time.hour] = min(min_duration[start_time.hour], diff)
					max_duration[start_time.hour] = max(max_duration[start_time.hour], diff)
				else:
					durations[start_time.hour] += start_time.minute
					min_duration[start_time.hour] = min(min_duration[start_time.hour], start_time.minute)
					max_duration[start_time.hour] = max(max_duration[start_time.hour], start_time.minute)
			elif i == delta - 1:
				durations[start_time.hour] += end_time.minute
				min_duration[start_time.hour] = min(min_duration[start_time.hour], end_time.minute)
				max_duration[start_time.hour] = max(max_duration[start_time.hour], end_time.minute)
			else:
				durations[start_time.hour] += 60
				min_duration[start_time.hour] = min(min_duration[start_time.hour], 60)
				max_duration[start_time.hour] = 60
			counts[start_time.hour] += 1
			start_time += timedelta(hours=1)

	# 把未赋值的最大/最小时长都换成0
	min_duration = list(map(lambda x: 0 if x == float('inf') else x, min_duration))
	max_duration = list(map(lambda x: 0 if x == float('-inf') else x, max_duration))

	# 保存成DataFrame
	result = {}
	result['hours'] = hours
	result['counts'] = counts
	result['durations'] = durations
	result['min_durations'] = min_duration
	result['max_durations'] = max_duration
	# tmp_durations = np.array(durations) - np.array(min_duration) - np.array(max_duration)
	# result['mean_durations'] = np.round(np.divide(tmp_durations, days - 2), 2)
	# 计算平均超时时长，并将0除以0得到的nan替换为0
	mean_durations = np.round(np.divide(durations, counts))
	nan_index = np.isnan(mean_durations)
	mean_durations[nan_index] = 0
	result['mean_durations'] = mean_durations

	result_df = pd.DataFrame(result)
	columns = ['hours', 'counts', 'durations', 'min_durations', 'max_durations', 'mean_durations']
	result_df.to_csv(d_path + unit, index=None, columns=columns)


# 根据超时事件在每个小时发生的次数和时长为每个设备计算报警规则
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
		counts = df.loc[idx]['counts']
		# 超时次数小于10次的忽略不计
		if counts < 10:
			rules[idx] = 5
		# 超时次数大于10次的按平均超时时长制定报警机制
		else:
			mean_duration = df.loc[idx]['mean_durations']
			if mean_duration != 0 and mean_duration%5 == 0:
				rules[idx] = int(mean_duration)
			else:
				rules[idx] = int( (mean_duration // 5 + 1)*5 )

	rules_df = {}
	rules_df['hours'] = [i for i in range(24)]
	rules_df['rules'] = rules
	rules_df = pd.DataFrame(rules_df)
	rules_df.to_csv(d_path + unit, index=None)


# 合并规则：报警规则类似的设备使用同一套规则
# 【
# 	unit：单元设备地址；
#  	o_path: 报警规则路径；
#  	d_path：合并后报警规则路径；
#  】


# 计算根据新的规则报警率下降情况
# 【
# 	unit：单元设备地址；
#  	before_path: 原超时报警数据路径；
#  	df_rules：新报警规则；
#  	df_classes: 设备的类别；
#  】
# return: 每个设备运用新规则后报警降低率
def alarm_reduce_rate(unit, before_path, df_rules, df_classes):
	before = before_path + unit
	o = open(before, 'rb')
	df_before = pd.read_csv(o, parse_dates=['start_time', 'end_time'], date_parser=dateparse2)

	# 若没有超时发生，返回0
	count_before = len(df_before)
	if count_before == 0:
		return 0

	# 找出设备所属类别
	the_class = ''
	classes = df_classes['class'].as_matrix()
	devices = df_classes['devices'].as_matrix()
	for i in range(len(devices)):
		if unit.split('.')[0] in devices[i]:
			the_class = classes[i]
			break

	# 找出类别对应规则
	rule = df_rules.loc[the_class].as_matrix()
	# print(the_class, rule)

	count_after = 0
	for idx in df_before.index:
		duration = df_before.loc[idx]['duration']

		# 大于1小时的情况都会报警
		if duration >= 1:
			count_after += 1
			continue

		# 判断小于一小时的情况有没有超出规则中的时长（规则中的单位为分钟），最多跨两个不同的时刻
		start_time, end_time = df_before.loc[idx]['start_time'], df_before.loc[idx]['end_time']
		if start_time.hour < end_time.hour:
			if 60 - start_time.minute >= rule[start_time.hour] or end_time.minute >= rule[end_time.hour]:
				count_after += 1
		elif end_time.minute - start_time.minute >= rule[start_time.hour]:
			count_after += 1

	# reduce_rate = format((count_before - count_after) / count_before, '.2%')
	reduce_rate = (count_before - count_after) / count_before
	# print(count_before, count_after, reduce_rate )
	return reduce_rate
