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


# 根据异常检测结果提取出表现正常的每小时数据;
# 【
# 	unit：单元设备地址；
#  	anomalies_path: 异常数据路径；
#  	timeout_path：超时数据路径；
#  	days_path：按天划分的数据集，用来获取该设备共有多少天的数据；
#  	d_path：结果存储的路径；
#  】
def hourly_open_to_close_time(unit, timeout_path, d_path):
	path = timeout_path + unit
	o = open(path, 'rb')
	timeout_df = pd.read_csv(o, parse_dates=['start_time', 'end_time'], date_parser=dateparse2)

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


# 将小时数据结合成分时段数据(按作息习性分成8个时间段)
def timeSlot_open_to_close_time(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	timeout_df = pd.read_csv(o, encoding='gbk')

	# 左开右闭区间
	timeSlots = ['0-5', '5-7', '7-10', '10-11', '11-13', '13-17','17-20', '20-24']
	durations = [0] * 8
	counts = [float('-inf')] * 8

	for idx in timeout_df.index:
		hour = timeout_df.loc[idx]['hours']
		count = timeout_df.loc[idx]['counts']
		duration = timeout_df.loc[idx]['durations']
		if hour >= 0 and hour <= 5:
			durations[0] += duration
			counts[0] = max(counts[0], count)   # 时段中超时事件发生的次数为时段中分小时发生超时事件次数的最大值
		elif hour <= 7:
			durations[1] += duration
			counts[1] = max(counts[1], count)
		elif hour <= 10:
			durations[2] += duration
			counts[2] = max(counts[2], count)
		elif hour <= 11:
			durations[3] += duration
			counts[3] = max(counts[3], count)
		elif hour <= 13:
			durations[4] += duration
			counts[4] = max(counts[4], count)
		elif hour <= 17:
			durations[5] += duration
			counts[5] = max(counts[5], count)
		elif hour <= 20:
			durations[6] += duration
			counts[6] = max(counts[6], count)
		else:
			durations[7] += duration
			counts[7] = max(counts[7], count)

	# 构造结果
	result = {}
	result['time_slot'] = timeSlots
	result['counts'] = counts
	result['durations'] = durations

	counts = list(map(lambda x: 0 if x == float('-inf') else x, counts))
	mean_durations = np.round(np.divide(durations, counts))
	nan_index = np.isnan(mean_durations)
	mean_durations[nan_index] = 0
	result['mean_durations'] = mean_durations

	result_df = pd.DataFrame(result)
	columns = ['time_slot', 'counts', 'durations', 'mean_durations']
	result_df.to_csv(d_path + unit, index=None, columns=columns)


# 根据超时事件在每个小时发生的次数和时长为每个设备计算报警规则
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def timeSlot_unit_alarm_rules(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o)
	rules = [0]*8

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

	columns = ['time_slot', 'rules']
	timeSlots = ['0-5', '5-7', '7-10', '10-11', '11-13', '13-17', '17-20', '20-24']
	rules_df = {}
	rules_df['time_slot'] = timeSlots
	rules_df['rules'] = rules
	rules_df = pd.DataFrame(rules_df)
	rules_df.to_csv(d_path + unit, columns=columns, index=None)
