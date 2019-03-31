#使用scipy的层次聚类函数进行聚类，并画出谱系聚类图和聚类结果图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import distance,linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering
from dateparsers import dateparse4
from datetime import datetime

# 计算距离矩阵并画出层次聚类图
# 【
#  	source_path: 源数据路径；
#  	result_path：结果存储的路径；
#  】
def plt_cluster(source_path, result_path):
	# 参数初始化
	path = source_path  # 开门不同时段
	o = open(path, 'rb')
	data = pd.read_csv(o, index_col='neighbor')

	# 生成点与点之间的距离矩阵,这里用的欧氏距离:
	disMat = distance.pdist(data, metric='euclidean')
	Z = linkage(disMat, method='average')  # 进行层次聚类:
	P=dendrogram( Z )											# 将层级聚类结果以树状图表示出来并保存
	plt.savefig( result_path + 'plot_dendrogram.png')
	# plt.show()


# 聚类，并保存结果
# 【
# 	n: 聚类簇的数量
# 	data: 原数据帧
# 	Z：距离矩阵
# 】
def cluster(n, source_path, result_path):
	# 读取数据
	path = source_path  # 开门不同时段
	o = open(path, 'rb')
	data = pd.read_csv(o, index_col='neighbor')

	# 聚类
	model= AgglomerativeClustering(n_clusters=n, linkage='ward')
	model.fit(data)
	cluster = model.labels_
	k = len(np.unique(cluster))									# 聚类簇的数量

	#详细输出原始数据及其类别
	if not os.path.exists(result_path + 'imgs/'):
		os.makedirs(result_path + 'imgs/')
	if not os.path.exists(result_path + 'csv/'):
		os.makedirs(result_path + 'csv/')

	r = pd.concat([data, pd.Series(cluster, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
	r.columns = list(data.columns) + [u'聚类类别']              #重命名表头

	plt.rcParams['font.sans-serif'] = ['SimHei']                #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False                  #用来正常显示负号

	style = ['ro-', 'go-', 'bo-', 'co-', 'mo-', 'yo-', 'ro-', 'go-', 'bo-', 'co-', 'mo-', 'yo-']
	xlabels = [u'工作日工作时段', u'工作日通勤时段', u'周末日间', u'凌晨']
	pic_output = result_path + 'imgs/type_'                 	#聚类图文件名前缀

	# 逐一作图，作出不同样式
	for i in range(k):
		plt.figure(figsize=(16,9))
		tmp = r[r[u'聚类类别'] == i].iloc[:,:4]                    #提取每一类除最后一列（label）的数据
		tmp.to_csv( result_path + 'csv/类别%s.csv' %(i+1) )     		#将每一类存成一个csv文件

		for j in range(len(tmp)):
			plt.plot( range(1, 5), tmp.iloc[j], style[i] )
			plt.xticks( range(1, 5), xlabels, rotation = 20 )         #坐标标签
		plt.title( u'门洞类别%s' % (i+1) )                         	#从1开始计数
		plt.subplots_adjust( bottom=0.15 )                        #调整底部
		plt.savefig( u'%s%s.png' % (pic_output, i+1) )            	#保存图片
		plt.close()


# 根据聚类结果聚合各类别小区数据（一天中每小时）
# 【
# 	type: 聚类结果类别名（如：类别1.csv）；
# 	type_path: 存储聚类结果的路径；
# 	counts_path：存储与聚类结果每个类对应的开门次数数据路径；
# 	d_path：聚合结果存储路径；
# 】
def clusters_counts_daily(type, type_path, counts_path, d_path):
	path = type_path + type
	o = open(path, 'rb')
	df = pd.read_csv(o, encoding='gbk')

	period = pd.period_range('2019-03-16', periods=10, freq='D').tolist()
	dates = [date.strftime('%Y-%m-%d') for date in period]

	for idx in df.index:
		neighbor = df.loc[idx]['neighbor']
		o_path = counts_path + neighbor + '/'
		units = os.listdir(o_path)
		df_neighbor = None
		for unit in units:
			o = open(o_path + unit, 'rb')
			df_unit = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse4)

			# 填充缺失日期的值为0
			if len(df_unit) < len(dates):
				for i in range(len(dates)):
					df_dates = list(df_unit['received_time'].map(lambda x: x.strftime('%Y-%m-%d')))
					if dates[i] not in df_dates:
						new_line = [pd.datetime.strptime(dates[i], '%Y-%m-%d')] + [0] * 24
						df_unit.loc[i] = new_line

			tmp = df_unit['received_time']
			del df_unit['received_time']
			if df_neighbor is None:
				df_neighbor = df_unit
			else:
				df_neighbor += df_unit

		df_neighbor = (df_neighbor / len(df)).round(2)		# 求平均
		df_neighbor.insert(0, 'received_time', tmp)
		csvFile = open(d_path + type, 'w')
		df_neighbor.to_csv(d_path + type, index=None)


# 根据聚类结果聚合各类别小区数据（一周中每天）
# 【
# 	type: 聚类结果类别名（如：类别1.csv）；
# 	type_path: 存储聚类结果的路径；
# 	counts_path：存储与聚类结果每个类对应的开门次数数据路径；
# 	d_path：聚合结果存储路径；
# 】
def clusters_counts_weekly(type, type_path, counts_path, d_path):
	path = type_path + type
	o = open(path, 'rb')
	df = pd.read_csv(o, encoding='gbk')

	period = pd.period_range('2019-03-16', periods=3, freq='7D').tolist()
	dates = [date.strftime('%Y-%m-%d') for date in period]

	for idx in df.index:
		neighbor = df.loc[idx]['neighbor']
		o_path = counts_path + neighbor + '/'
		units = os.listdir(o_path)
		df_neighbor = None
		for unit in units:
			o = open(o_path + unit, 'rb')
			df_unit = pd.read_csv(o, parse_dates=['received_time'], date_parser=dateparse4)

			# 填充缺失日期的值为0
			if len(df_unit) < len(dates):
				for i in range(len(dates)):
					df_dates = list(df_unit['received_time'].map(lambda x: x.strftime('%Y-%m-%d')))
					if dates[i] not in df_dates:
						new_line = [pd.datetime.strptime(dates[i], '%Y-%m-%d')] + [0] * 7
						df_unit.loc[i] = new_line

			tmp = df_unit['received_time']
			del df_unit['received_time']
			if df_neighbor is None:
				df_neighbor = df_unit
			else:
				df_neighbor += df_unit

		df_neighbor = (df_neighbor / len(df)).round(2)		# 求平均
		df_neighbor.insert(0, 'received_time', tmp)
		csvFile = open(d_path + type, 'w')
		df_neighbor.to_csv(d_path + type, index=None)
