from pandas import DataFrame
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import datetime
import csv
import os
import re
from dateparsers import dateparse1, dateparse2, dateparse3, dateparse4

# -------------------------------------------------------- 自定义函数  ------------------------------------------------------- #

# 计算单元设备每小时/每日开门次数
# 【
# 	address: 单元设备地址；
# 	unit：小区地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def rooms_open_counter(address, unit, o_path, d_path):
	path = o_path + unit + '/' + address
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2)
	df_out = pd.DataFrame(columns = ["received_time", 'open_count'])
	i = 0
	end_time = None
	for idx in df.index:
		curr_time = df.loc[idx]['received_time']
		if not end_time:
			end_time = curr_time
			open_frequency = 0

		if d_path.split('/')[-2] == '1day':
			if end_time.day != curr_time.day:
				df_out.loc[i] = {'received_time': end_time.strftime('%Y-%m-%d'), 'open_count': open_frequency}
				open_frequency = 0
				end_time = curr_time
				i += 1

			if idx == df.index[-1]:
				df_out.loc[i] = {'received_time': end_time.strftime('%Y-%m-%d'), 'open_count': open_frequency}

		else:
			if end_time.hour != curr_time.hour:
				df_out.loc[i] = {'received_time': end_time, 'open_count': open_frequency}
				open_frequency = 0
				end_time = curr_time
				i += 1

			if idx == df.index[-1]:
				df_out.loc[i] = {'received_time': end_time, 'open_count': open_frequency}

		if df.loc[idx]['status'] == 4:
			open_frequency += 1

	new_path = d_path + unit + '/'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	csvFile = open(new_path + address, 'w')
	df_out.to_csv(new_path + address, index=None, encoding='utf-8')


# 将整天都没有开门情况超过5天的设备剔除保存，注意保存无开门天数
# 【
# 	address: 单元设备地址；
# 	unit：小区地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def delete_little_open(address, unit, o_path, d_path):
	path = o_path + unit + '/' + address
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse4)
	open = df['open_count'].values
	zero_count = len(open) - len(open.nonzero())
	if zero_count > 5:
		new_path = d_path + unit + '/'
		if not os.path.exists(new_path):
			os.makedirs(new_path)
		csvFile = open(new_path + address, 'w')
		df.to_csv(new_path + address, index=None)
	return zero_count


# 计算单元设备每日漏报次数(omission_requency)、误报次数（未发出警报却出现解除警报状态）
def omissions_and_error(unit):
	path = 'units/' + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = datepars1e)
	address = unit.split('.')[0]
	df_out = pd.DataFrame(columns = ["received_time", 'omission', 'error'])
	i = 0
	end_time = None
	for idx in df.index:
		curr_time = df.loc[idx]['received_time']
		if not end_time:
			end_time = curr_time
			relieve, timeout = 0, 0

		if end_time.day != curr_time.day:
			if timeout >= relieve:
				omission, error = timeout - relieve, 0
			else:
				error, omission = relieve - timeout, 0
			df_out.loc[i] = {'received_time': end_time, 'omission': omission, 'error': error }
			relieve, timeout = 0, 0
			end_time = curr_time
			i += 1

		if idx == df.index[-1]:
			df_out.loc[i] = {'received_time': end_time, 'omission': omission, 'error': error }

		if df.loc[idx]['status'] == 6:
			timeout += 1
		elif df.loc[idx]['status'] == 7:
			relieve += 1

	csvfile = open('counts/omissions_and_error/' + unit, 'w')
	df_out.to_csv('counts/omissions_and_error/' + unit)

# 计算单元设备总漏报次数(omission_requency)、误报次数（未发出警报却出现解除警报状态）
def total_omissions_and_error(unit, df_out, i):
	path = 'counts/omissions_and_error/' + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse1)
	address = unit.split('.')[0]
	df_out.loc[i] = {"address": address, 'omission': df['omission'].sum(), 'error': df['error'].sum()}


# 计算单元设备每日开门次数open_frequency、漏报次数(omission_requency)、超时报警次数（正常解除报警的情况）timeout_frequency
def snythesize_counter(unit):
	path = 'units/' + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse1)
	address = unit.split('.')[0]
	df_out = pd.DataFrame(columns = ["received_time", 'open', 'omission', 'timeout'])
	i = 0
	end_time = None
	for idx in df.index:
		curr_time = df.loc[idx]['received_time']
		if not end_time:
			end_time = curr_time
			open_frequency, relieve, timeout = 0, 0, 0

		if end_time.day != curr_time.day:
			if timeout >= relieve:
				timeout_frequency, omission_requency = relieve, timeout - relieve
			else:
				timeout_frequency, omission_requency = timeout, timeout - relieve
			df_out.loc[i] = {'received_time': end_time, 'open': open_frequency, 'omission': omission_requency, 'timeout': timeout_frequency }
			open_frequency, relieve, timeout = 0, 0, 0
			end_time = curr_time
			i += 1

		if idx == df.index[-1]:
			df_out.loc[i] = {'received_time': end_time, 'open': open_frequency, 'omission': omission_requency, 'timeout': timeout_frequency }

		if df.loc[idx]['status'] == 4:
			open_frequency += 1
		elif df.loc[idx]['status'] == 6:
			timeout += 1
		elif df.loc[idx]['status'] == 7:
			relieve += 1

	csvfile = open('counts/snythesize/' + unit, 'w')
	df_out.to_csv('counts/snythesize/' + unit)


# 计算单元设备总开门次数open_frequency、漏报次数(omission_requency)、超时报警次数（正常解除报警的情况）timeout_frequency
def total_snythesize_counter(unit, df_out, i):
	path = 'units/' + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2)
	address = unit.split('.')[0]
	open_frequency, relieve, timeout = 0, 0, 0
	for idx in df.index:
		if df.loc[idx]['status'] == 4:
			open_frequency += 1
		elif df.loc[idx]['status'] == 6:
			timeout += 1
		elif df.loc[idx]['status'] == 7:
			relieve += 1

		if idx == df.index[-1]:
			if timeout >= relieve:
				timeout_frequency, omission_requency = relieve, timeout - relieve
			else:
				timeout_frequency, omission_requency = timeout, timeout - relieve
			df_out.loc[i] = {'address': address, 'open': open_frequency, 'omission': omission_requency, 'timeout': timeout_frequency }


# 计算单元设备总开门次数(工作日工作时段，工作日夜间，周末日间，凌晨)
# 【
# 	unit：单元设备地址；
# 	df_out: 结果DataFrame；
# 	i：结果DataFrame索引；
#  	o_path: 源数据路径；
#  】
def total_open_condition(address, unit, o_path, df_out, i):
	path = o_path + unit + '/' + address
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2)
	workday_workhours, workday_commuting_hours, weekend_daytime, wee_hours = 0, 0, 0, 0
	for idx in df.index:
		received_time = df.loc[idx]['received_time']
		if df.loc[idx]['status'] == 4:
			# 凌晨
			if received_time.hour >=0 and received_time.hour <= 4:
				wee_hours += 1
			# 周末日间
			elif received_time.weekday() == 5 or received_time.weekday() == 6:
				weekend_daytime += 1
			# 工作日工作时间
			elif received_time.hour >= 10 and received_time.hour <= 17:
				workday_workhours += 1
			# 工作日通勤时间
			else:
				workday_commuting_hours += 1

		if idx == df.index[-1]:
			df_out.loc[i] = { "neighbor": unit, 'address':address.split('.')[0], 'workday_workhours': workday_workhours, 'workday_commuting_hours': workday_commuting_hours, 'weekend_daytime': weekend_daytime, 'wee_hours':wee_hours }


# 计算每个设备每个小时正常开门的总次数并转化为CSV文件（0时到1时之间的算作1时，以此类推）
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def open_frequency_peer_hour(address, unit, o_path, d_path):
	path = o_path + unit + '/' + address
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2, encoding='gbk')
	df_out = pd.DataFrame(columns = ['received_time', "1时", "2时", "3时", "4时", "5时", "6时", "7时", "8时", "9时", "10时", "11时", "12时", "13时", "14时", "15时", "16时", "17时", "18时", "19时", "20时", "21时", "22时", "23时", "24时"])
	i = 0
	curr_date = None
	hours = [0]*24
	for idx in df.index:
		if curr_date is None:
			curr_date = df.loc[idx]['received_time']
			hours[ curr_date.hour ] += df.loc[idx]['open_count']
		elif curr_date.strftime('%Y-%m-%d') == df.loc[idx]['received_time'].strftime('%Y-%m-%d'):
			hours[ df.loc[idx]['received_time'].hour] += df.loc[idx]['open_count']
		else:
			df_out.loc[i] = {'received_time': curr_date.strftime('%Y-%m-%d'), "1时": hours[0], "2时": hours[1], "3时": hours[2], "4时": hours[3], "5时": hours[4], "6时": hours[5], "7时": hours[6], "8时": hours[7], "9时": hours[8],
						     "10时": hours[9], "11时": hours[10], "12时": hours[11], "13时": hours[12], "14时": hours[13], "15时": hours[14], "16时": hours[15], "17时": hours[16], 
						     "18时": hours[17], "19时": hours[18], "20时": hours[19], "21时": hours[20], "22时": hours[21], "23时": hours[22], "24时": hours[23]}
			hours = [0]*24
			curr_date = df.loc[idx]['received_time']
			hours[ curr_date.hour ] += df.loc[idx]['open_count']
			i += 1
		if idx == df.index[-1]:
			df_out.loc[i] = {'received_time': curr_date.strftime('%Y-%m-%d'), "1时": hours[0], "2时": hours[1], "3时": hours[2], "4时": hours[3], "5时": hours[4], "6时": hours[5], "7时": hours[6], "8时": hours[7], "9时": hours[8],
						     "10时": hours[9], "11时": hours[10], "12时": hours[11], "13时": hours[12], "14时": hours[13], "15时": hours[14], "16时": hours[15], "17时": hours[16], 
						     "18时": hours[17], "19时": hours[18], "20时": hours[19], "21时": hours[20], "22时": hours[21], "23时": hours[22], "24时": hours[23]}

	new_path = d_path + unit + '/'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	csvFile = open(new_path + address, 'w')
	df_out.to_csv(new_path + address, index=None, encoding='utf-8')


# 计算每个设备每个小时正常开门的总次数并转化为CSV文件（0时到1时之间的算作1时，以此类推）
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def open_frequency_peer_day(address, unit, o_path, d_path):
	path = o_path + unit + '/' + address
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse4, encoding='gbk')
	df_out = pd.DataFrame(columns = ['received_time', "周一", "周二", "周三", "周四", "周五", "周六", "周日"])
	i = 0
	start_date = None
	days = [0]*7
	for idx in df.index:
		curr_date = df.loc[idx]['received_time']
		counts = df.loc[idx]['open_count']
		if start_date is None:
			start_date = df.loc[idx]['received_time']
			days[ curr_date.weekday() ] += counts
		elif curr_date.weekday() != 6:
			days[ curr_date.weekday() ] += counts
		else:
			df_out.loc[i] = {'received_time':start_date, "周一": days[0], "周二": days[1], "周三": days[2], "周四": days[3], "周五": days[4], "周六": days[5], "周日": days[6]}
			days = [0]*7
			start_date = df.loc[idx]['received_time']
			days[ start_date.weekday() ] += counts
			i += 1

		if idx == df.index[-1]:
			df_out.loc[i] = {'received_time':start_date, "周一": days[0], "周二": days[1], "周三": days[2], "周四": days[3], "周五": days[4], "周六": days[5], "周日": days[6]}

	new_path = d_path + unit + '/'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	csvFile = open(new_path + address, 'w')
	df_out.to_csv(new_path + address, index=None, encoding='utf-8')

# 将逐小时数据按天分开成单独的csv文件，方便绘图
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def split_by_day(unit, o_path, d_path):
	new_path = d_path + unit.split('.')[0] + '/'
	if not os.path.exists(new_path):
		os.makedirs( new_path )
	
	path = o_path + unit
	o1 = open(path, 'rb')
	o2 = open(path, 'rb')
	df1 = pd.read_csv(o1, parse_dates=['received_time'], date_parser = dateparse3, encoding='gbk')
	df2 = pd.read_csv(o2, parse_dates=['received_time'], date_parser = dateparse2, encoding='gbk')
	groups = df2.groupby(df1['received_time'])
	for group in groups:
	    group[1].to_csv( new_path + str(group[0])[:10] + '.csv', index=False, encoding='utf-8')


# 将逐天数据按周分开成单独的csv文件，方便绘图
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def split_by_week(unit, o_path, d_path):
	new_path = d_path + unit.split('.')[0] + '/'
	if not os.path.exists(new_path):
		os.makedirs( new_path )
	
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse4, encoding='gbk')
	start_date = None
	df_out = pd.DataFrame(columns = ['received_time', 'open_count'])
	i = 0
	for idx in df.index:
		curr_date = df.loc[idx]['received_time']
		counts = df.loc[idx]['open_count']
		if start_date is None:
			start_date = df.loc[idx]['received_time']
			df_out.loc[i] = {'received_time': start_date, 'open_count': counts}
			i += 1
		elif df.loc[idx]['received_time'].weekday() != 6:
			df_out.loc[i] = {'received_time': curr_date, 'open_count': counts}
			i += 1
		elif df.loc[idx]['received_time'].weekday() == 6:
			df_out.loc[i] = {'received_time': curr_date, 'open_count': counts}
			csvfile = open(new_path + curr_date.strftime('%Y-%m-%d') + '.csv', 'w')
			df_out.to_csv(new_path + curr_date.strftime('%Y-%m-%d') + '.csv')
			i = 0
			start_date = None
			df_out = pd.DataFrame(columns = ['received_time', 'open_count'])


# 提取出春节期间正常开门次数数据
def spring_festival_open(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	if o_path.split('/')[-2] == '1day':
		df = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse4)
	else:
		df = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse2)

	start = pd.datetime.strptime('2019-02-04 00:00', '%Y-%m-%d %H:%M')
	end = pd.datetime.strptime('2019-02-19 00:00', '%Y-%m-%d %H:%M')
	for idx in df.index:
		date = df.loc[idx]['received_time']
		if date < start or date > end:
			df.drop(idx, inplace=True)
	csvFile = open(d_path + unit, 'w')
	df.to_csv(d_path + unit, index=None)


# 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长
def open_to_close_time(unit):
	path = 'units/' + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=["received_time"], date_parser=dateparse1)
	df_out = pd.DataFrame(columns = ['start_time', 'end_time', 'duration'])
	i, idx = 0, 0			# 控制原df的遍历，控制新df_out的新行添加

	# 遍历查找事件的开始时间（超时的前五分钟-正常开门）和结束时间（报警解除）
	while idx < len(df):
		start_time, end_time = None, None
		if df.loc[idx]['status'] == 6:
			start_time = df.loc[idx - 1]['received_time'] - timedelta(minutes = 5)
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

	new_path = 'counts/open_to_close_time/' + unit
	csvfile = open(new_path, 'w')
	df_out.to_csv(new_path)


# -----------------------------------------------  以下函数调用按实际需求调用  ------------------------------------------------ #

# 计算单元设备每日漏报次数(omission_requency)、误报次数（未发出警报却出现解除警报状态）
# units = os.listdir('units/')
# if not os.path.exists('counts/omissions_and_error/'):
# 	os.makedirs( 'counts/omissions_and_error/' )
# for unit in units:
# 	omissions_and_error(unit)

# 计算单元设备总漏报次数(omission_requency)、误报次数（未发出警报却出现解除警报状态）
# df_out = pd.DataFrame(columns = ["address", 'omission', 'error'])
# i = 0
# units = os.listdir('counts/omissions_and_error/')
# for unit in units:
# 	total_omissions_and_error(unit, df_out, i)
# 	i += 1
# csvfile = open('counts/total_omissions_and_error.csv', 'w')
# df_out.to_csv('counts/total_omissions_and_error.csv')

# 计算单元设备每日开门次数open_frequency、漏报次数(timeout - relieve)、超时报警次数（正常解除报警的情况）relieve
# units = os.listdir('units/')
# if not os.path.exists('counts/snythesize/'):
# 	os.makedirs( 'counts/snythesize/' )
# for unit in units:
# 	snythesize_counter(unit)

# 计算单元设备总的开门次数、漏报次数、超时报警次数（正常解除报警的情况）
# units = os.listdir('units/')
# df_out = pd.DataFrame(columns = ["address", 'open', 'omission', 'timeout'])
# i = 0
# for unit in units:
# 	total_snythesize_counter(unit, df_out, i)
# 	i += 1
# csvfile = open('counts/total_snythesize.csv', 'w')
# df_out.to_csv('counts/total_snythesize.csv')


# 计算每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件的时长
# if not os.path.exists('counts/open_to_close_time/'):
# 	os.makedirs('counts/open_to_close_time/')
# units = os.listdir('units/')
# for unit in units:
# 	open_to_close_time(unit)

