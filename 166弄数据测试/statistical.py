from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import os
from dateparsers import dateparse4

# 读取数据
def read_df(unit, o_path, d_path):
	path = o_path + unit
	o = open(path, 'rb')
	df = pd.read_csv(o, parse_dates=['received_time'], date_parser = dateparse4)

	# 根据情况过滤出一日模型的工作日/周末数据(一周模型无需过滤)
	status = d_path.split('/')[-2]
	if status == 'daily_workday':
		for idx in df.index:
			curr_date = df.loc[idx]['received_time']
			if curr_date.weekday() == 5 or curr_date.weekday() == 6:
				df.drop(idx, inplace=True)
	elif status == 'daily_weekend':
		for idx in df.index:
			curr_date = df.loc[idx]['received_time']
			if curr_date.weekday() != 5 and curr_date.weekday() != 6:
				df.drop(idx, inplace=True)
	address = unit.split('.')[0]
	return(df, address)

# 建模
# 【
# 	unit：单元设备地址；
#  	o_path: 源数据路径；
#  	d_path：结果存储的路径；
#  】
def model(unit, o_path, d_path):
	# 读取数据
	df, address = read_df(unit, o_path, d_path)

	# 建模
	i = 0
	model_df = DataFrame(columns = ['received_time', 'l_bound', 'mean', 'u_bound'])
	for col in df.columns[1:]:
		mean_val = int( df[col].mean() )
		std_val = int( df[col].std() )
		u_bound = mean_val + std_val
		l_bound = mean_val - std_val if (mean_val - std_val)>0 else 0
		model_df.loc[i] = { 'received_time': col, 'l_bound': l_bound, 'mean': mean_val, 'u_bound': u_bound  }
		i += 1

	# 保存模型数据
	new_path = d_path + address + '/'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	csv_file = open(new_path + unit, 'w')
	model_df.to_csv(new_path + unit, index=None)

	# 输出模型图
	fig, ax = plt.subplots(figsize=(16,9))
	# model_df['received_time'] = model_df['received_time'].map(lambda x: x[:-1])
	l1, = ax.plot( model_df['received_time'], model_df['mean'], 'go-', label='最佳情况')
	l2, = ax.plot( model_df['received_time'], model_df['u_bound'], 'co--', label='上界')
	l3, = ax.plot( model_df['received_time'], model_df['l_bound'], 'co--', label='下界')
	plt.legend([l1, l2, l3], ['最佳情况', '上界', '下界'])
	plt.rcParams['font.sans-serif'] = ['SimHei'] 				#用来正常显示中文标签
	ax.set_xlabel('时间')
	ax.set_ylabel('总次数')
	fig.suptitle(address)
	plt.savefig(new_path + address + '.png')
	plt.close()

