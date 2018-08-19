# -*- coding:utf-8 -*-
from __init__ import *
from basic import *

def calculate_error_rate(gx, label, D):
	error_rate = np.sum(np.array(gx ^ label).astype(np.int) * D)
	return error_rate

def calculate_alpha(error_rate):
	alpha = 1 / 2 * np.log((1 - error_rate) / error_rate)
	return alpha

def update_sample_weight(D, alpha, gx, label):
	label_product_gx = np.array(gx ^ label).astype(np.int)  # 相同为0，不同为1
	label_product_gx[np.where(label_product_gx == 1)] = -1
	label_product_gx[np.where(label_product_gx == 0)] = 1

	z = np.sum(D * np.exp(-alpha * label_product_gx))
	updated_D = D / z * np.exp(-alpha * label_product_gx)
	updated_D = updated_D.reshape(len(updated_D), 1)
	return updated_D

def sigmoid(x):
	s = 1 / (1 + np.exp(-x))
	return s

def ensemble(models, feats, label, Encoder=True):
	alphas, gxs = [], []
	D = np.array([1] * len(label)) / len(label)  # 初始化权重
	D = D.reshape(len(D), 1)
	for model in models:  # model, ss, fi = models[0], sss[0], fis[0]
		if Encoder:
			gx = autoencoder_predict(model, feats) # 模型预测输出
		else:
			gx= clf_predict(model, feats)

		gxs.append(gx)
		error_rate = calculate_error_rate(gx, label, D)  # 计算错误率
		alpha = calculate_alpha(error_rate)  # 计算alpha
		alphas.append(alpha)
		D = update_sample_weight(D, alpha, gx, label)  # 更新权重

	fx = alphas[0] * gxs[0]
	for alpha, gx in zip(alphas[1:], gxs[1:]):
		fx += alpha * gx
	# Gx = sigmoid(temp.reshape(len(temp), 1))
	return fx, alphas, gxs

def vote(fx):
	voted = []
	for x in fx:
		if x >= 0:
			voted.append(1)
		else:
			voted.append(0)
	voted = np.array(voted).reshape(len(voted), 1)
	return voted

if __name__ == '__main__':

	grouped_model_names = rand_group6x9(False)
	group_alphas = []  # 每组模型的alphas都保存在里面
	for index, model_names in enumerate(grouped_model_names):
		print (model_names)

		models = load_models(model_names)  # 载入第i组的模型
		print (models)
		feats, label = load_OneGroupOfdata(model_names)  # 载入第i组的模型对应的全部数据和标签
		print ('Load data successfully...')
		fx, alphas, gxs = ensemble(models, feats, label, True)  # 模型集成
		group_alphas.append(alphas)
		# 投票器，有5组模型预测该数据为真则集成模型为真，否者为假
		Gx = gxs[0]
		for gx in gxs[1:]:
			Gx += gx
		voted = vote(Gx)
		# 打印投票器的准确率
		print('acc:', accuracy_score(voted, label))
		print('or:', OR(voted.astype(int), label.astype(int)))
		print('len_group_alphas: ', len(group_alphas))

	save_pkl(group_alphas, 'group_alphas.pkl')


