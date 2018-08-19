from __init__ import *
from basic import *
from PIL import Image

def reshape_image(image_array):    #reshape
    rows, cols = image_array.shape
    label = []
    for col in range(cols):
        for row in range(rows):
            label.append(image_array[row][col])
    label = np.array(label).reshape((rows * cols), 1)
    return label

def label2image(label):  #将数据还原成原来图像的格式
    image = label * 255
    image = Image.fromarray(image.reshape((200, 200)).astype('uint8').T)
    return image


def save_image(label, save_path):
    image = label2image(label)
    image.save(save_path)
    # print('save ' + save_path + ' success!')

class AdaBoostBinaryClassifier(object):
	'''
	INPUT:
	- n_estimator (int)
	  * The number of estimators to use in boosting
	  * Default: 50
	- learning_rate (float)
	  * Determines how fast the error would shrink
	  * Lower learning rate means more accurate decision boundary,
		but slower to converge
	  * Default: 1
	'''

	def __init__(self,
				 n_estimators=50,
				 learning_rate=1):

		self.base_estimator = DecisionTreeClassifier(max_depth=1)
		self.n_estimator = n_estimators
		self.learning_rate = learning_rate

		# Will be filled-in in the fit() step
		self.estimators_ = []
		self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

	def fit(self, x, y):
		'''
		INPUT:
		- x: 2d numpy array, feature matrix
		- y: numpy array, labels
		Build the estimators for the AdaBoost estimator.
		'''
		sample_weight = np.ones(x.shape[0])/x.shape[0]
		for tree in range(self.n_estimator):
			estimator, sample_weight, estimator_weight= \
				self._boost(x,y, sample_weight)
			self.estimators_.append(estimator)
			self.estimator_weight_[tree]=estimator_weight



	def _boost(self, x, y, sample_weight):
		'''
		INPUT:
		- x: 2d numpy array, feature matrix
		- y: numpy array, labels
		- sample_weight: numpy array
		OUTPUT:
		- estimator: DecisionTreeClassifier
		- sample_weight: numpy array (updated weights)
		- estimator_weight: float (weight of estimator)
		Go through one iteration of the AdaBoost algorithm. Build one estimator.
		'''
		estimator = clone(self.base_estimator)
		dtc = estimator
		dtc.fit(x, y, sample_weight=sample_weight)
		pred_y = dtc.predict(x)
		indicator = np.ones(x.shape[0])*[pred_y!=y][0]
		err = np.dot(sample_weight, indicator) / np.sum(sample_weight)
		alpha = np.log((1-err)/err)
		new_sample_weight = sample_weight* np.exp(alpha*indicator)
		return estimator, new_sample_weight, alpha



	def predict(self, x):
		'''
		INPUT:
		- x: 2d numpy array, feature matrix
		OUTPUT:
		- labels: numpy array of predictions (0 or 1)
		'''
		predicts = []
		for estimator in self.estimators_:
			pred = estimator.predict(x)
			pred[pred==0] = -1
			predicts.append(pred)

		predicts = np.array(predicts)

		pr = np.sign(np.dot(self.estimator_weight_, predicts))
		pr[pr==-1] = 0
		return pr


	def score(self, x, y):
		'''
		INPUT:
		- x: 2d numpy array, feature matrix
		- y: numpy array, labels
		OUTPUT:
		- score: float (accuracy score between 0 and 1)
		'''
		#accuracy = TP+TN / n
		predictions = self.predict(x)
		n= x.shape[0]
		tp = np.sum(predictions * y)
		tn = np.sum((1-predictions)* (1-y))
		acc = (tp+tn)/n
		return acc

	def sklearn_AdaBoostClassifier(self, x_train, y_train, x_test, y_test):
		model = AdaBoostClassifier(self.base_estimator, self.n_estimator)
		model.fit(x_train, y_train)
		return model.score(x_test, y_test)

if __name__ == '__main__':
	classfier = AdaBoostClassifier(n_estimators=50,learning_rate=1)
	# model_ids = gen_model_ids(root=data_path)
	model_ids = ['512','513','522','201','225','223','410','302','305','507','509','508','501','506','314',
	             '511', '519', '202', '409', '232', '228', '220', '227', '308', '306', '502', '307']
	print (len(model_ids))
	grouped_model_id = model_ids[:2]
	# print (grouped_model_id)
	print (len(grouped_model_id))

	train_feats, train_label = load_group_data(grouped_model_id)
	# train_feats, train_label = select_points(train_feats, train_label, num_points=168800)
	# train_feats, train_label = select_points(train_feats, train_label, num_points=189000)

	# train_feats, train_label = load_data('202')
	train_label = train_label.ravel()
	print(train_label.shape)

	# print (train_feats.shape, train_label.shape)
	classfier.fit(train_feats, train_label)

	joblib.dump(classfier, 'model.pkl', compress=1)
	print ('Save model successfully! ')

	# ORs = []
	# ids = []
	for model_id in ['511']:
		feats, label = load_data(model_id)
		predict = classfier.predict(feats)
		predict = predict.reshape(len(predict), 1)
		# print (predict.shape)

		iou = OR(predict.astype(int), label.astype(int))
		if iou > 0.5:
			print (' test ' + 'data' + model_id + ': ' + str(iou))
			image_name = 'test_data_' + model_id + '.bmp'
			save_image(predict.astype(int), './predict_images7/'+image_name)

			# ORs.append(iou)
			# ids.append(model_id)

	
		# if iou > 0.5:
		# 	ORs.append(iou)
		# 	ids.append(model_id)


		

	# print (np.mean(ORs))
	# for i in ids:
	# 	print (i)


