#使用scipy的层次聚类函数进行聚类，并画出谱系聚类图和聚类结果图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import distance,linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering

# 计算距离矩阵并画出层次聚类图
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
	return data

# 聚类，并保存结果
# n: 聚类簇的数量
# data: 原数据帧
# Z：距离矩阵
def cluster(n, data, result_path):
	model= AgglomerativeClustering(n_clusters=n, linkage='ward') 			#根据linkage matrix Z得到聚类结果
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

	for i in range(1, k+1):  #逐一作图，作出不同样式
		plt.figure(figsize=(16,9))
		tmp = r[r[u'聚类类别'] == i].iloc[:,:4]                    #提取每一类除最后一列（label）的数据
		tmp.to_csv( result_path + 'csv/类别%s.csv' %(i) )     		#将每一类存成一个csv文件

		for j in range(len(tmp)):                                 #作图
			plt.plot( range(1, 5), tmp.iloc[j], style[i - 1] )
			plt.xticks( range(1, 5), xlabels, rotation = 20 )         #坐标标签
			plt.title( u'门洞类别%s' % (i) )                         	#从1开始计数
			plt.subplots_adjust( bottom=0.15 )                        #调整底部
			plt.savefig( u'%s%s.png' % (pic_output, i) )            	#保存图片
			plt.close()
