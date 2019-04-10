# 1. 读取原文件，并将其按地址划分为csv文件
# 2. 删除重复数据：同一设备同一状态之间间隔在15s以内则删去重复者
# 3. 补充数据：补充在一次超时事件中漏报的状态（正常开门，超市未关门报警，报警解除）

import pandas as pd
from datetime import timedelta
from dateparsers import dateparse2

# 1. 读取所有原文件，并聚合为同一个文件
# 【
# 	file_name: 原文件名；
#  	column_names: 需要解析的列名；
# 	o_path：源文件路径；
#  】
#  返回：读取的DataFrame
def gather_data(file_name, column_names, usecols , o_path):
	path = o_path + file_name
	# o = open(path, 'rb')
	df = pd.read_excel(path, names = column_names )
	df = df.loc[:,usecols ]
	return df

# 读取原始文件并按地址划分为单独的文件
# 【
#  	usecols:数据分析需要的列；
# 	o_path：源文件路径；
#  	d_path：结果存储的路径；
#  】
def split_by_address( usecols, o_path, d_path):
	df = pd.read_excel(o_path, usecols = usecols)
	df =  df[df['app_name'] == '住宅单元门体状态监测报警']
	del df['app_name']
	# print(df.head())

	# 按地址划分
	groups = df.groupby(df['address'])
	for group in groups:
		group[1].to_csv(d_path + str(group[0]) + '.csv', index=False, encoding='utf-8')


# 删除同一设备在15秒内同一状态的数据
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def delete_redundancy(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse2)

	idx = 0
	while idx < len(df) - 1:
		curr_time = df.loc[idx]['received_time']
		curr_status = df.loc[idx]['status']
		next_time = df.loc[idx + 1]['received_time']
		next_status = df.loc[idx + 1]['status']
		if curr_status == next_status and next_time - curr_time <= timedelta(seconds=15):
			df.drop(idx + 1, inplace=True)
			df = df.reset_index(drop = True)
		idx += 1
	csvFile = open(d_path + unit, 'w')
	df.to_csv(d_path + unit, index=None, encoding='utf-8')

# 数据状态补充
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def data_supplement(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse2)
	address = unit.split('.')[0]
	df_insert = pd.DataFrame(columns = ['address','status','received_time'])
	idx = 0			# 控制原df的遍历

	# 检查报警前无“正常开门”和缺少“报警解除”状态的情况并补充
	while idx < len(df):
		if df.loc[idx]['status'] == 6:			# “超时未关门报警”
			if df.loc[idx - 1]['status'] != 4:	# 若超时的前一个状态不是“正常开门”状态，在超时前5分钟插入一个正常开门状态
				time = df.loc[idx]['received_time'] - timedelta(minutes = 5)
				df_insert.loc[0] = {'address': address,'status': 4,'received_time': pd.datetime.strftime(time,'%Y-%m-%d %H:%M:%S')}
				df = pd.concat([df[:idx], df_insert, df[idx:]], axis = 0)
				df = df.reset_index(drop = True)		# 重置索引
				idx += 1
			idx += 1
			for j in range(idx, len(df)):
				if df.loc[j]['status'] == 0:	# 若超时的下一个状态为“开门状态”，继续
					continue
				elif df.loc[j]['status'] == 7:	# 找到“报警解除”状态，一次完整事件
					idx = j + 1
					break
				else:							# 不是“开门状态”也不是“报警解除”，说明未上报“报警解除”状态，在上一个状态（超时/开门状态）后5分钟补充上“报警解除”
					time = df.loc[j-1]['received_time'] + timedelta(minutes = 5)
					df_insert.loc[0] = {'address': address,'status': 7,'received_time': pd.datetime.strftime(time,'%Y-%m-%d %H:%M:%S')}
					df = pd.concat([df[:j], df_insert, df[j:]], axis = 0)
					df = df.reset_index(drop = True)		# 重置索引
					idx = j + 2
					break
			else:
				idx += 1
		else:
			idx += 1
	df = df.reset_index(drop = True)			# 重置索引

	# 检查缺少“超时未关门报警”状态的情况并补充
	idx = 0
	i = 0
	while idx < len(df):
		if df.loc[idx]['status'] == 7:			# “报警解除”
			if idx == 0:						# 第一个状态就是解除，则在其前5分钟加上超时状态
				time = df.loc[idx]['received_time'] - timedelta(minutes = 5)
				df_insert.loc[0] = {'address': address,'status': 6,'received_time': pd.datetime.strftime(time,'%Y-%m-%d %H:%M:%S')}
				df = pd.concat([df_insert, df], axis = 0)
				df = df.reset_index(drop = True)		# 重置索引
				idx = idx + 1
			else:
				for j in range(idx - 1, -1, -1):
					if df.loc[j]['status'] == 0:	# 若报警解除的上一个状态为“开门状态”，继续
						continue
					elif df.loc[j]['status'] == 6:	# 找到“超时未关门报警”状态，一次完整事件
						idx = idx + 1
						break
					elif df.loc[j]['status'] == 4:	# 未出现“超时未关门报警”即向前找到了“正常开门”状态“，在正常开门后5分钟加上”超时未关门报警“状态
						time = df.loc[j]['received_time'] + timedelta(minutes = 5)
						df_insert.loc[0] = {'address': address,'status': 6,'received_time': pd.datetime.strftime(time,'%Y-%m-%d %H:%M:%S')}
						df = pd.concat([df[:j+1], df_insert, df[j+1:]], axis = 0)
						df = df.reset_index(drop = True)		# 重置索引
						idx = idx + 1
						break
				if j == 0:
					idx += 1
		else:
			idx += 1
	df = df.reset_index(drop = True)		# 重置索引
	csvFile = open(d_path + unit, 'w')
	df.to_csv(d_path + unit, index = None, encoding='utf-8')
