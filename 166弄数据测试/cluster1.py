#使用sklearn的DBSCAN聚类函数进行聚类，并画出谱系聚类图和聚类结果图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

def cluster1(source_path, result_path):
	#参数初始化

	path = source_path 	#开门不同时段
	o = open(path, 'rb')
	data = pd.read_csv(o, index_col = 'address' )
	X = data.values

	# 手肘法调参
	eps_list = list(range(300, 3000, 100))
	SC = []
	for i in eps_list:
		db = DBSCAN(eps=i, min_samples=1).fit(X)
		labels = db.labels_
		n_labels = len(set(labels))
		if n_labels == 1:
			break
		else:
			SC.append(metrics.silhouette_score(X, labels))
	best_eps = eps_list[np.argmax(SC)]
	# Compute DBSCAN
	db = DBSCAN(eps=best_eps, min_samples=1).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	k = len(set(labels))									# 聚类簇的数量

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = [plt.cm.Spectral(each)
			  for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = [0, 0, 0, 1]

		class_member_mask = (labels == k)

		xy = X[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
				 markeredgecolor='k', markersize=14)

		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
				 markeredgecolor='k', markersize=6)

	plt.title('Estimated number of clusters: %d' % k)
	plt.savefig('../clusters/聚类结果.png')

	#详细输出原始数据及其类别
	if not os.path.exists(result_path + 'imgs/'):
		os.makedirs(result_path + 'imgs/')
	if not os.path.exists(result_path + 'csv/'):
		os.makedirs(result_path + 'csv/')

	r = pd.concat([data, pd.Series(labels, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
	r.columns = list(data.columns) + [u'聚类类别']              #重命名表头

	plt.rcParams['font.sans-serif'] = ['SimHei']                #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False                  #用来正常显示负号

	style = ['ro-', 'go-', 'bo-', 'co-', 'mo-', 'yo-']
	xlabels = [u'工作日工作时段', u'工作日通勤时段', u'周末日间', u'凌晨']
	pic_output = result_path + 'imgs/type_'                 		#聚类图文件名前缀

	for i in range(k):  #逐一作图，作出不同样式
		plt.figure()
		tmp = r[r[u'聚类类别'] == i].iloc[:,:4]                    	#提取每一类除最后一列（label）的数据
		tmp.to_csv( result_path + 'csv/类别%s.csv' %(i) )     		#将每一类存成一个csv文件

		for j in range(len(tmp)):                                 	#作图
			plt.plot( range(1, 5), tmp.iloc[j], style[i - 1] )

			plt.xticks( range(1, 5), xlabels, rotation = 20 )         #坐标标签
			plt.title( u'门洞类别%s' % (i) )                         	#从1开始计数
			plt.subplots_adjust( bottom=0.15 )                        #调整底部
			plt.savefig( u'%s%s.png' % (pic_output, i) )            	#保存图片
