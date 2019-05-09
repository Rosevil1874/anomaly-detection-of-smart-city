#!/usr/bin/env python
# coding: utf-8
# 找出异常日期
import pandas as pd
from scipy.fftpack import fft, ifft
from dateparsers import dateparse2, dateparse4
import scipy.signal as signal

# 找出异常日期并保存结果
# 【
# 	unit：单元设备地址；
#  	o_o_path: 源数据路径；
#  	d_o_path：结果存储的路径；
#  】
def anomaly_detection(unit, o_path, d_path):
	o = open(o_path + unit , 'rb')
	if o_path.split('/')[-2] == '1day':
		df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse4, encoding='gbk')
	else:
		df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2, encoding='gbk')
	address = unit.split('.')[0]

	# 1. 快速傅里叶变化(周期)
	# 转化为ndarray
	s = df['open_count'].as_matrix()
	# 消除直流分量
	# 标准化
	mean = s.mean()
	std = s.std()
	s = (s - mean) / std
	# s = (s - s.mean())/s.std()
	# 快速傅里叶变换
	main_df = ifft ( fft(s) )

	# 2. 滑动中值（趋势）
	medfilted = signal.medfilt(s, 11)

	# 3. 原始数据减去周期效应和趋势(滑动中值)得到误差项
	error = s - main_df- medfilted

	# 根据偏差确定异常
	# mean = error.mean()
	# std = error.std()
	mean = s.mean()
	std = s.std()
	up_bound = mean + 2.5*std
	lower_bound = mean - 2.5*std

	df_out = pd.DataFrame(columns = ['received_time', 'error_val', 'open_count', 'abnormal_type'])
	i = 0
	for idx in range(len(s)):
		val = s[idx]
		if val > up_bound :
			if o_path.split('/')[-2] == '1day':
				df_out.loc[i] = {'received_time': df.loc[idx]['received_time'].strftime('%Y-%m-%d'), 'error_val': error[idx],
								 'open_count': df.loc[idx]['open_count'], 'abnormal_type': 1}
			else:
				df_out.loc[i] = {'received_time': df.loc[idx]['received_time'], 'error_val': error[idx],'open_count':
					df.loc[idx]['open_count'], 'abnormal_type': 1}
			i += 1
		elif val < lower_bound:
			if o_path.split('/')[-2] == '1day':
				df_out.loc[i] = {'received_time': df.loc[idx]['received_time'].strftime('%Y-%m-%d'), 'error_val': error[idx],
								 'open_count': df.loc[idx]['open_count'], 'abnormal_type': 0}
			else:
				df_out.loc[i] = {'received_time': df.loc[idx]['received_time'], 'error_val': error[idx],
								 'open_count': df.loc[idx]['open_count'], 'abnormal_type': 0}
			i += 1
	if i != 0:
		csvfile = open(d_path + address + '.csv', 'w')
		df_out.to_csv(d_path + address + '.csv')


#  日期异常检测：使用均值方差模型验证异常检测正确性
# 输入：
# 【
# 	device_o_path：要判断异常检测正确率的设备路径，
# 	weekly_o_path：一周均值方差模型路径，
# 	corr_anomalies_o_path：异常检测结果正确的条目保存路径
# 】
# 返回：【corr_rate：异常检测正确率】
def daily_anomaly_corr_rate(device_path, weekly_path, corr_anomalies_path):
	unit = device_path.split('/')[-1]
	o = open(weekly_path + unit.split('.')[0] +  '/' + unit)
	statistical_weekly = pd.read_csv(o, index_col=['received_time'], encoding='utf-8')
	o = open(device_path)
	anomaly = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse4)

	corr_count, i = 0, 0
	df_corr = pd.DataFrame(columns = ['received_time', 'error_val', 'open_count', 'abnormal_type'])
	for idx in anomaly.index:
		week = anomaly.loc[idx]['received_time'].weekday()
		open_count = anomaly.loc[idx]['open_count']
		weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
		u_bound = statistical_weekly.loc[weekdays[week]]['u_bound']
		l_bound = statistical_weekly.loc[weekdays[week]]['l_bound']
		if open_count > u_bound or open_count < l_bound:
			corr_count += 1
			df_corr.loc[i] = anomaly.loc[idx]
			i += 1
	csvFile = open(corr_anomalies_path + unit, 'w')
	df_corr.to_csv(corr_anomalies_path + unit)

	corr_rate = corr_count / len(anomaly)
	return  corr_rate

#  日期+时刻异常检测：使用均值方差模型判断异常检测正确性
#  输入：
# 【
# 	device_o_path：要判断异常检测正确率的设备路径，
# 	workday_o_path：工作日均值方差模型路径，
# 	weekend_o_path：周末均值方差模型路径，
# 	corr_anomalies_o_path：异常检测结果正确的条目保存路径
# 】
# 返回：【corr_rate：异常检测正确率】
def hourly_anomaly_corr_rate(device_o_path, workday_o_path, weekend_o_path, corr_anomalies_o_path):
	address = device_o_path.split('/')[-1]
	o = open(workday_o_path + address.split('.')[0] + '/' + address)
	statistical_workday = pd.read_csv(o, index_col=['received_time'])
	o = open(weekend_o_path + address.split('.')[0] + '/' + address)
	statistical_weekend = pd.read_csv(o, index_col=['received_time'])
	o = open(device_o_path)
	anomaly = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2, encoding='gbk')

	corr_count, i = 0, 0
	df_corr = pd.DataFrame(columns = ['received_time', 'error_val', 'open_count', 'abnormal_type'])
	for idx in anomaly.index:
		week = anomaly.loc[idx]['received_time'].weekday()
		hour = anomaly.loc[idx]['received_time'].hour
		open_count = anomaly.loc[idx]['open_count']
		#     周末
		if week ==5 or week == 6:
			u_bound = statistical_weekend.loc[str(hour + 1) + '时']['u_bound']
			l_bound = statistical_weekend.loc[str(hour + 1) + '时']['l_bound']
			if open_count > u_bound or open_count < l_bound:
				corr_count += 1
				df_corr.loc[i] = anomaly.loc[idx]
				i += 1

		#     工作日
		else:
			u_bound = statistical_workday.loc[str(hour + 1) + '时']['u_bound']
			l_bound = statistical_workday.loc[str(hour + 1) + '时']['l_bound']
			if open_count > u_bound or open_count < l_bound:
				corr_count += 1
				df_corr.loc[i] = anomaly.loc[idx]
				i += 1

	csvFile = open(corr_anomalies_o_path + address, 'w')
	df_corr.to_csv(corr_anomalies_o_path + address)
	corr_rate = corr_count / len(anomaly)
	return corr_rate

# 将异常结果汇总成一个表，并加上地址、周几字段
#  输入：
# 【
# 	unit：需要合并的设备，
# 	o_path：需要合并的设备路径，
# 】
# 返回：【corr_rate：异常检测正确率】
def gather_anomalies(unit, o_path):
	path = o_path + unit
	o = open(path, 'rb')

	# 分情况解析日期
	if o_path.split('/')[-3] == 'daily':
		df = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse4)
	else:
		df = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse2)

	# 添加地址、周几、日期类型（周末，工作日、节假日）字段
	del df['Unnamed: 0']
	df['address'] = unit.split('.')[0]
	df['weekday'] = ''
	df['day_type'] = ''
	weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
	for i in df.index:
		date = df.loc[i]['received_time']
		weekday = date.weekday()
		df.loc[i,'weekday'] = weekdays[weekday]

		# 判断日期属性，节假日包括（元旦，五一，国庆，圣诞）
		month, day = date.month, date.day
		if month == 1 and day == 1 or month == 5 and day >= 1 and day <= 3 or month == 10 and day >= 1 and day <= 7 or month == 12 and day == 25:
			df.loc[i,'day_type'] = '节假日'
		elif weekday == 5 or weekday == 6:
			df.loc[i,'day_type'] = '周末'
		else:
			df.loc[i,'day_type'] = '工作日'


	# 如果是时刻异常检测，再加上异常发生的时间段字段
	if o_path.split('/')[-3] == 'hourly':
		df['period_of_time'] = ''
		for i in df.index:
			hour = df.loc[i]['received_time'].hour
			if hour >=0 and hour <= 4:
				df.loc[i,'period_of_time'] = '凌晨'
			elif hour <= 8:
				df.loc[i,'period_of_time'] = '早上'
			elif hour <= 10:
				df.loc[i,'period_of_time'] = '早通勤'
			elif hour <= 17:
				df.loc[i,'period_of_time'] = '工作时段'
			elif hour <= 19:
				df.loc[i,'period_of_time'] = '晚通勤'
			else:
				df.loc[i,'period_of_time'] = '晚上'


	# 将地址和周几字段移到最前面
	cols = df.columns.values.tolist()
	cols.insert(1, cols.pop(cols.index('weekday')))
	cols.insert(0, cols.pop(cols.index('address')))
	df = df.ix[:, cols]
	return df

