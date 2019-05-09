# 根据不同需求将字符串格式的时间解析为时间戳
import pandas as pd

def dateparse1(timestamp):
	target_date = pd.datetime.strptime(timestamp,'%Y.%m.%d %H:%M:%S')
	return target_date

def dateparse2(timestamp):
	target_date = pd.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')
	return target_date

def dateparse3(timestamp):
	target_date = pd.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')
	return target_date.strftime('%Y-%m-%d')

def dateparse4(timestamp):
	target_date = pd.datetime.strptime(timestamp,'%Y-%m-%d')
	return target_date
