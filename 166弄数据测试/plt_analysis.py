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
from dateparsers import  dateparse1, dateparse2, dateparse3, dateparse4


# ---------------------------------------------- 根据不同需求解析时间戳  ---------------------------------------------------  #
#
# # 字符串格式时间解析为时间戳
# def dateparse(received_time):
# 	target_date = pd.datetime.strptime(received_time,'%Y/%m/%d %H:%M:%S')
# 	return target_date
#
# # 字符串格式时间解析为时间戳
# def dateparse2(received_time):
# 	target_date = pd.datetime.strptime(received_time,'%Y-%m-%d %H:%M:%S')
# 	return target_date
#
# # 字符串格式时间解析为时间戳
# def dateparse3(received_time):
# 	target_date = pd.datetime.strptime(received_time,'%Y-%m-%d %H:%M:%S')
# 	return target_date.strftime('%Y-%m-%d')
#
# # 字符串格式时间解析为时间戳
# def dateparse4(received_time):
# 	target_date = pd.datetime.strptime(received_time,'%Y-%m-%d')
# 	return target_date
#
# # 字符串格式时间解析为时间戳
# def dateparse5(received_time):
# 	target_date = pd.datetime.strptime(received_time,'%Y/%m/%d %H:%M')
# 	return target_date.strftime('%Y-%m-%d')
#

# -------------------------------------------------------- 自定义函数  ------------------------------------------------------- #

# 画出每个设备一天中每个小时对应的开门次数
def day_img(door):
	dates = os.listdir('counts/peer_day/' + door)
	for date in dates:
		print_day_img(door, date)

def print_day_img(door, date):
	new_path = 'imgs/peer_day/' + door + '/'
	if not os.path.exists(new_path):
		os.makedirs( new_path )

	path = 'counts/peer_day/' + door + '/' + date
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2)
	del df['Unnamed: 0']
	fig, ax = plt.subplots()
	line, = ax.plot(df['received_time'], df[door], 'g-' , label = door)
	ax.xaxis.set_major_formatter(mdate.DateFormatter('%H'))
	ax.set_xlabel('时刻')
	ax.set_ylabel('正常开门次数')
	fig.suptitle(date.split('.')[0])
	# plt.xticks(rotation=90)
	plt.grid(True)
	plt.savefig('imgs/peer_day/' + door + '/' + date.split('.')[0] + '.png')
	plt.close(fig)
	# plt.show()

# 画出每个设备一周中每天对应的开门次数
def week_img(door):
	dates = os.listdir('counts/peer_week/' + door)
	for date in dates:
		print_week_img(door, date)

def print_week_img(door, date):
	new_path = 'imgs/peer_week/' + door + '/'
	if not os.path.exists(new_path):
		os.makedirs( new_path )

	path = 'counts/peer_week/' + door + '/' + date
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse4)
	del df['Unnamed: 0']
	fig, ax = plt.subplots()
	line, = ax.plot(df['received_time'], df['counts'], 'g-' , label = door)
	ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
	ax.set_xlabel('日期')
	ax.set_ylabel('正常开门次数')
	fig.suptitle(date.split('.')[0])
	plt.xticks(rotation=30)
	plt.grid(True)
	plt.savefig('imgs/peer_week/' + door + '/' + date.split('.')[0] + '.png')
	plt.close(fig)
	# plt.show()

# 画出每个设备工作日/周末的一天开门次数模型
def workday_peer_hour(door):
	path = 'counts/peer_hour/' + door
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['日期'], date_parser = dateparse4, encoding='gbk')
	df.rename(columns = {'日期':'received_time'}, inplace=True)
	del df['Unnamed: 0']
	for idx in df.index:
		curr_date = df.loc[idx]['received_time']
		if curr_date.weekday() == 5 or curr_date.weekday() == 6:
			df.drop(idx)
	plt.figure(figsize=(16,9))
	plt.rcParams['font.sans-serif'] = ['SimHei']                #正常显示中文标签
	df.boxplot()
	plt.savefig('imgs/分时段模型图/workday/' + door.split('.')[0] + '.png')
	# plt.clf()
	# plt.show()

# 画出每个设备一周开门次数模型
def week_peer_day(door):
	path = 'counts/peer_day/' + door
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['日期'], date_parser = dateparse3, encoding='gbk')
	df.rename(columns = {'日期':'received_time'}, inplace=True)
	del df['Unnamed: 0']
	plt.figure(figsize=(16,9))
	plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
	df.boxplot()
	plt.savefig('imgs/one_week/' + door.split('.')[0] + '.png')
	# plt.clf()
	# plt.show()

# 删除节假日以外的图
def holiday_peer_hour(door):
	path = 'imgs/holiday/' + door + '/'
	imgs = os.listdir(path)
	for img in imgs:
		img_name = img.split('.')[0]
		if img_name != '2018-12-25' and img_name != '2019-01-01' and img_name != '2019-02-04' and img_name != '2019-02-05':
			os.remove('imgs/holiday/' + door + '/' + img)

# 每个设备的漏报、误报次数比较
def omission_and_error_analysis():
	path = 'counts/total_omissions_and_error.csv'
	o = open(path, 'rb')
	df = pd.read_csv(o, encoding='gbk')
	fig, ax = plt.subplots()
	fig.suptitle('每个设备的漏报、误报次数比较', fontsize=14)
	l1, = ax.plot(df['address'], df['omission'], 'co-' , label = '漏报次数')
	l2, = ax.plot(df['address'], df['error'], 'mo-' , label = '误报次数')
	plt.legend([l1, l2], ['漏报次数', '误报次数'])
	ax.set_xlabel('设备地址')
	ax.set_ylabel('总次数')
	plt.xticks(rotation = 90)
	plt.show()

# 画出每个设备一次超时事件开门时长(1小时及以内，1~3小时，3~6小时，6~10，10小时以上)占比
def plt_open_to_close_time(door):
	path = 'counts/open_to_close_time/' + door
	o = open(path, 'rb')
	df = pd.read_csv(o, encoding='gbk')

	# 计算每个时长的事件个数
	counts = [0]*5
	for idx in df.index:
		duration = df.loc[idx]['duration']
		if duration <= 1:
			counts[0] += 1
		elif duration <= 3:
			counts[1] += 1
		elif duration <= 6:
			counts[2] += 1
		elif duration <= 10:
			counts[3] += 1
		else:
			counts[4] +=1

	# 画出对应饼图并保存
	labels = ['1小时及以内', '1~3小时', '3~6小时', '6~10小时', '10小时以上']
	exist_labels = []
	exist_counts = []
	# 过滤空数据
	for i in range(len(counts)):
		if counts[i] != 0:
			exist_labels.append(labels[i])
			exist_counts.append(counts[i])
	fig, ax = plt.subplots(figsize = (16, 9))
	ax.pie(exist_counts, labels = exist_labels, autopct='%1.1f%%', startangle=90)
	fig.suptitle(door.split('.')[0] + '：总' + str(len(df)) + '次')
	# plt.show()
	new_path = 'imgs/open_to_close_time/'
	plt.savefig(new_path + door.split('.')[0] + '.png')

	return counts

# 计算所有设备一次超时事件开门时长占比并画出饼图			
def plt_total_open_to_close_time(total_counts):
	labels = ['1小时及以内', '1~3小时', '3~6小时', '6~10小时', '10小时以上']
	fig, ax = plt.subplots(figsize = (16, 9))
	ax.pie(total_counts, labels = labels, autopct='%1.1f%%', startangle=90)
	fig.suptitle('总超时事件次数：' + str( sum(total_counts)) + '次')
	# plt.show()
	plt.savefig('imgs/total_open_to_close_time.png')	

# 画出每个设备春节期间每天开门次数图：
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def plt_spring_days(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse4)
	address = unit.split('.')[0]

	fig, ax = plt.subplots(figsize=(16, 9))
	l, = ax.plot(df['received_time'], df['open_count'], 'go-')
	fig.suptitle(address)
	ax.set_xlabel('日期')
	ax.set_ylabel('开门次数')
	plt.savefig(d_path + address + '.png')
	plt.close(fig)


# ------------------------------------------------ 以下函数调用按实际需求调用  ------------------------------------------------ #


# 画出每个设备一天中每个小时对应的开门次数
# doors = os.listdir('counts/peer_day/')
# for door in doors:
# 	day_img(door)	

# 画出每个设备一周中每天对应的开门次数
# doors = os.listdir('counts/peer_week/')
# for door in doors:
# 	week_img(door)

# 将每天的逐小时开门次数调整为工作日/周末逐小时开门次数
# doors = os.listdir('counts/peer_hour/')
# for door in doors:
# 	workday_peer_hour(door)

# 一周开门次数模型
# doors = os.listdir('counts/peer_day/')
# for door in doors:
# 	week_peer_day(door)

# 删除节假日以外的图
# doors = os.listdir('imgs/holiday/')
# for door in doors:
# 	holiday_peer_hour(door)

# 每个设备的漏报、误报次数比较
# omission_and_error_analysis()

# 画出每个设备一次超时事件开门时长占比
# if not os.path.exists('imgs/open_to_close_time/'):
# 	os.makedirs('imgs/open_to_close_time/')
# doors = os.listdir('counts/open_to_close_time/')
# total_counts = [0]*5
# for door in doors:
# 	counts = plt_open_to_close_time(door)
# # 	计算所有设备一次超时事件开门时长占比并画出饼图
# 	total_counts = list(np.array(total_counts) + np.array(counts))
# plt_total_open_to_close_time(total_counts)


# 画出设备正常范围和异常曲线
# path1 = 'statistical_model/daily_workday/岚皋路166弄42号/岚皋路166弄42号.csv'
# path2 = 'counts/peer_day_split/岚皋路166弄42号/2019-02-04.csv'
# o1 = open(path1, 'rb')
# o2 = open(path2, 'rb')
# df1 = pd.read_csv(o1)
# df2 = pd.read_csv(o2, parse_dates = ['received_time'], date_parser = dateparse2)
#
# # 补充真实数据到24小时
# del df2['Unnamed: 0']
# df2.reset_index(drop = True)
# i = 0
# while i < len(df2):
# 	time = df2.loc[i]['received_time']
# 	insert = pd.DataFrame(columns = ['received_time', '岚皋路166弄42号'])
# 	if time.hour != i:
# 		insert.loc[0] = {'received_time':time - timedelta(hours = 1), '岚皋路166弄42号':0}
# 		df2 = pd.concat([df2[:i], insert, df2[i:]], axis = 0)
# 		df2 = df2.reset_index(drop = True)
# 	i += 1
#
# fig, ax = plt.subplots()
# df1['hour'] = df1['hour'].map(lambda x: x[:-1])
# l1, = ax.plot( df1['hour'], df1['mean'], 'go-', label='最佳情况')
# l2, = ax.plot( df1['hour'], df1['u_bound'], 'co--', label='上界')
# l3, = ax.plot( df1['hour'], df1['l_bound'], 'co--', label='下界')
# l4, = ax.plot( df1['hour'], df2['岚皋路166弄42号'], 'ro--', label='实际值' )
# plt.legend([l1, l2, l3, l4], ['最佳情况', '上界', '下界', '实际值'])
# plt.rcParams['font.sans-serif'] = ['SimHei'] 				#用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False                  #用来正常显示负号
# ax.set_xlabel('时刻')
# ax.set_ylabel('总次数')
# fig.suptitle('岚皋路166弄42号 2月4日')
# # plt.savefig(new_path + address + '.png')
# plt.show()
