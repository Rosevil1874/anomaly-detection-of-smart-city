import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

import warnings
warnings.filterwarnings('ignore')

# 导入数据
def read_data_train(path):
	path = 'counts/auto_encoder/' + path + '.csv'
	o = open(path, 'rb')
	df = pd.read_csv(o, encoding='gbk')
	del df['Unnamed: 0']
	del df['日期']
	# 按8：2划分训练集和测试集
	# data_train, data_test = train_test_split(df, test_size=0.2)
	return df


def train_and_test(data_train, Class, door, df_out, df_out_idx):
	o = open('counts/auto_encoder/all/' + Class + '/' + door, 'rb')
	new_data = pd.read_csv(o, encoding='gbk')
	del new_data['Unnamed: 0']
	address = door.split('.')[0]
	columns = new_data.columns
	data_test = new_data[ list(columns)[1:]]

	# 设置autoencoder参数
	input_dim = data_train.shape[1]	# 输入数据的维度
	encoding_dim = 32				# 隐藏层节点数分别为32，16，16，32
	num_epoch = 50
	batch_size = 8					# 调参得最佳batch size

	# 四层：激活函数分别为tanh, relu, tanh, relu
	input_layer = Input(shape=(input_dim, ))
	encoder = Dense(encoding_dim, activation='tanh',
					activity_regularizer=regularizers.l1(10e-5))(input_layer)
	encoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
	decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
	decoder = Dense(input_dim, activation='relu')(decoder)
	autoencoder = Model(inputs=input_layer, outputs=decoder)
	autoencoder.compile(optimizer='adam',
						loss = 'mean_squared_error',
						metrics=['mae'])

	# 设置模型保存路径，训练模型
	checkpointer = ModelCheckpoint(filepath='auto_encoder/class1/peer_hour_model.h5',
									verbose = 0,
									save_best_only=True)
	history = autoencoder.fit(data_train, data_train,
							  epochs=num_epoch,
	                          batch_size=batch_size,
	                          shuffle=True,
	                          validation_data=(data_test, data_test),
	                          verbose=1, 
	                          callbacks=[checkpointer]).history

	# 画出损失函数图
	# plt.figure(figsize=(16,9))
	# plt.subplot(121)
	# plt.plot(history['loss'], c='dodgerblue', lw=3)
	# plt.plot(history['val_loss'], c='coral', lw=3)
	# plt.title('model loss')
	# plt.ylabel('mse'); plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper right')

	# plt.subplot(122)
	# plt.plot(history['mean_absolute_error'], c='dodgerblue', lw=3)
	# plt.plot(history['val_mean_absolute_error'], c='coral', lw=3)
	# plt.title('model mae')
	# plt.ylabel('mae'); plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper right');

	# plt.show()


	# 读取模型并测试
	o = open('counts/auto_encoder/all/' + Class + '/' + door, 'rb')
	new_data = pd.read_csv(o, encoding='gbk')
	del new_data['Unnamed: 0']
	autoencoder = load_model('auto_encoder/class1/peer_hour_model.h5')
	columns = new_data.columns
	data_test = new_data[ list(columns)[1:]]		# 选取测试集中除“日期”外的列
	pred_test = autoencoder.predict(data_test)


	# 计算还原误差MSE和MAE
	mse_test = np.mean(np.power(data_test - pred_test, 2), axis=1)
	mae_test = np.mean(np.abs(data_test - pred_test), axis=1)
	# print(mse_test)
	# print(mae_test)
	for i in range(len(mae_test)):
		if mae_test[i] >= 3:
			df_out.loc[df_out_idx] = {'address': address, 'date': new_data.loc[i]['日期']}
			df_out_idx += 1
	mse_df = pd.DataFrame()
	# mse_df['Class'] = [0] * len(mse_test)
	mse_df['MSE'] = np.hstack([mse_test])
	mse_df['MAE'] = np.hstack([mae_test])
	mse_df = mse_df.sample(frac=1).reset_index(drop=True)

	return df_out_idx

# 分别画出测试集的还原误差MAE和MSE
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.scatter(mse_df.index, 
#             mse_df['MAE'],  
#             alpha=0.7, 
#             marker='o', 
#             c='dodgerblue', 
#             label='MAE')
# plt.title('Reconstruction MAE')
# plt.ylabel('Reconstruction MAE'); plt.xlabel('Index')
# plt.subplot(122)
# plt.scatter(mse_df.index, 
#             mse_df['MSE'],  
#             alpha=0.7, 
#             marker='^', 
#             c='coral', 
#             label='MSE')
# plt.title('Reconstruction MSE')
# plt.ylabel('Reconstruction MSE'); plt.xlabel('Index')
# plt.show()

# 画出Precision-Recall曲线
# plt.figure(figsize=(14, 6))
# for i, metric in enumerate(['MAE', 'MSE']):
#     plt.subplot(1, 2, i+1)
#     precision, recall, _ = precision_recall_curve([0] * len(data_test), mse_df[metric])
#     pr_auc = auc(recall, precision)
#     plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
#     plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
#     plt.xlabel('Recall'); plt.ylabel('Precision')
# plt.show()

# # 画出ROC曲线
# plt.figure(figsize=(14, 6))
# for i, metric in enumerate(['MAE', 'MSE']):
#     plt.subplot(1, 2, i+1)
#     fpr, tpr, _ = roc_curve([0] * len(data_test), mse_df[metric])
#     roc_auc = auc(fpr, tpr)
#     plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f'%(metric, roc_auc))
#     plt.plot(fpr, tpr, c='coral', lw=4)
#     plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
#     plt.ylabel('TPR'); plt.xlabel('FPR')
# plt.show()

classes = os.listdir('counts/auto_encoder/all/')
for Class in classes:
	data_train = read_data_train(Class)
	doors = os.listdir('counts/auto_encoder/all/' + Class + '/')
	df_out = pd.DataFrame(columns = ['address', 'date'])
	df_out_idx = 0
	for door in doors:
		print(door)
		df_out_idx = train_and_test(data_train, Class, door, df_out, df_out_idx)
	csvfile = open('counts/auto_encoder/result/' + Class + '.csv', 'w')
	df_out.to_csv('counts/auto_encoder/result/' + Class + '.csv')
