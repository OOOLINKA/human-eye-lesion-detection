# -*- coding:utf-8 -*-
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

def test(train_group_index, test_group_index):  #定义测试类，第一组模型对生成模型23456的剩余五组数据进行测试
    # grouped_model_names[0], group_alphas, 0, 1
    grouped_model_names = rand_group6x9(True)
    group_alphas = load_group_alphas()
    model_names = grouped_model_names[train_group_index]
    test_gxs = []
    test_feats, test_label = load_OneGroupOfdata(grouped_model_names[test_group_index])
    models = load_models(model_names)
    for model in models:  # model= models[0]...
        test_gx = autoencoder_predict(model, test_feats)  # 模型预测输出
        test_gxs.append(test_gx)
    alphas = group_alphas[train_group_index]
    test_fx = alphas[0] * test_gxs[0]
    for alpha, test_gx in zip(alphas[1:], test_gxs[1:]):
        test_fx += alpha * test_gx
    return test_fx, alphas, test_gxs, test_label


def test_all():
    all_gxs_dump_path = 'all_gxs.pkl'   #所有生成的预测值
    all_labels_dump_path = 'all_labels.pkl' #所有的真实值
    if os.path.exists(all_gxs_dump_path):
        print('all_gxs exist')
        with open(all_gxs_dump_path, 'rb') as f1:
            all_gxs = pickle.load(f1)
        with open(all_labels_dump_path, 'rb') as f2:
            all_labels = pickle.load(f2)
    else:
        print('all_gxs not exist')
        all_gxs = []  # 所有组的gxs
        all_labels = []  # 所有组对应的label
        for train_group_index in range(6):
            test_group_indexs = list(set(range(6)) - set([train_group_index]))
            group_gxs = []  # 第i组模型预测的所有gxs
            group_label = []
            for test_group_index in test_group_indexs:
                test_fx, alphas, test_gxs, test_label = test(train_group_index, test_group_index)
                group_gxs.append(test_gxs)
                group_label.append(test_label)
            all_gxs.append(group_gxs)
            all_labels.append(group_label)
        with open(all_gxs_dump_path, 'wb') as f1:
            pickle.dump(all_gxs, f1, protocol=4)
        with open(all_labels_dump_path, 'wb') as f2:
            pickle.dump(all_labels, f2, protocol=4)
    return all_gxs, all_labels

def calculate_precision(all_gxs, all_labels, all_alphas):   #定义计算精度这一指标
    all_precisions = []
    all_ORs = []
    for model_group in range(len(all_gxs)):
        model_precisions = []
        model_ORs = []
        print('model group: ' + str(model_group + 1))
        group_gxs = all_gxs[model_group]  # 模型X对所有组的预测
        group_label = all_labels[model_group]  # 模型X对应测试数据的标签
        group_alphas = all_alphas[model_group]

        test_groups_range = list(set(range(6)) - set([model_group]))

        for test_index, test_group in zip(range(len(group_gxs)), test_groups_range):
            group_precisions = []
            group_ORs = []
            groupX_gxs = group_gxs[test_index]  # 组X的预测结果
            groupX_label = group_label[test_index]  # 组X的标签
            gx1 = groupX_gxs[0].copy().astype(np.int)
            alpha1 = group_alphas[0].copy()
            gx1[np.where(gx1 == 0)] = -1
            # Gx = zero2neg1(gx1) * alpha1
            Gx = gx1 * alpha1

            for gx, alphas in zip(groupX_gxs[1:].copy(), group_alphas[1:].copy()):
                gx = gx.astype(np.int)
                gx[np.where(gx == 0)] = -1
                Gx += gx * alphas
            predict = np.sign(Gx)
            predict[np.where(predict == -1)] = 0
            voted_predict = predict

            # images = []
            # images_paths = []
            for image_num in range(9):
                image = voted_predict[image_num * 40000: (image_num + 1) * 40000, :]
                precision = accuracy_score(image, groupX_label[image_num * 40000: (image_num + 1) * 40000, :])
                group_precisions.append(precision)

                or_score = OR(image.astype(int), groupX_label[image_num * 40000: (image_num + 1) * 40000,:].astype(int))

                group_ORs.append(or_score)

                # images.append(image)
                image_name = 'model' + str(model_group + 1) + '_test_group' + str(test_group + 1) + '_img' + str(
                    image_num + 1) + '.bmp'
                path = 'predict_images/model' + str(model_group + 1) + '/test_group' + str(
                    test_group + 1) + '/' + image_name
                # save_image(image, path)

            voted_acc = accuracy_score(voted_predict, groupX_label)
            print('     ' + 'test group: ' + str(test_group + 1) + ' | accuracy_score: ' + str(voted_acc))

            voted_or = OR(voted_predict.astype(int), groupX_label.astype(int))
            print('     ' + 'test group: ' + str(test_group + 1) + ' | OR: ' + str(voted_or))

            model_precisions.append(group_precisions)
            model_ORs.append(group_ORs)
        all_precisions.append(model_precisions)
        all_ORs.append(model_ORs)
    return all_precisions, all_ORs


def print_precisions(all_precisions):

    for model_group, model_precisions in enumerate(all_precisions):
        merged_model_precisions = []
        print('model ' + str(model_group+1) + ' precisions: ')
        test_groups_range = list(set(range(6)) - set([model_group]))
        for group_precisions, test_groups in zip(model_precisions, test_groups_range):
            merged_model_precisions.extend(group_precisions)
            print('    group ' + str(test_groups+1) + ' precisions: ')
            f = lambda x: round(x, 3)
            temp = np.array(list(map(f, group_precisions)))
            print('          ', temp)
            print('           group_precisions mean: ', np.mean(temp))
            print('           group_precisions variance: ', np.var(temp))
        print('   model_precisions mean: ', np.mean(merged_model_precisions))
        print('   model_precisions variance: ', np.var(merged_model_precisions))

def print_all_ORs(all_ORs):
    for model_group, model_ORs in enumerate(all_ORs):
        merged_model_ORs = []
        print('model ' + str(model_group+1) + ' ORs: ')
        test_groups_range = list(set(range(6)) - set([model_group]))
        for group_ORs, test_groups in zip(model_ORs, test_groups_range):
            merged_model_ORs.extend(group_ORs)
            print('    group ' + str(test_groups+1) + ' ORs: ')
            f = lambda x: round(x, 10)
            temp = np.array(list(map(f, group_ORs)))
            print('          ', temp)
            print('           group_ORs mean: ', np.mean(temp))
            print('           group_ORs variance: ', np.var(temp))
        print('   model_ORs mean: ', np.mean(merged_model_ORs))
        print('   model_ORs variance: ', np.var(merged_model_ORs))

if __name__ == '__main__':
	all_gxs = load_pkl('all_gxs.pkl')
	all_labels = load_pkl('all_labels.pkl')
	# all_gxs, all_labels = test_all()
	all_alphas = load_pkl('group_alphas.pkl')

	all_precisions, all_ORs = calculate_precision(all_gxs, all_labels, all_alphas)
	print_precisions(all_precisions)
	print_all_ORs(all_ORs)


