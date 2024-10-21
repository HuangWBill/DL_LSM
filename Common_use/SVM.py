# -*- coding: utf-8 -*-
# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
'''
SVM模型训练并用该模型绘制滑坡易发性地图

输入：数据样本所在目录（Addfactors1_7.py生成的结果）/生成txt保存原始标签及预测标签位置/待预测影像存放位置/预测后易发性地图存放路径
     分块行、列大小/通道数/SVM参数gamma和C
输出：SVM预测的结果AUC及精度/SVM得到的滑坡易发性地图

注意：本程序选用RBF（径向基函数）
'''

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import sklearn
from Common_use import GRID
from Common_use import ROC_AUC as roc_s

v=sklearn.__version__
if int(v[2:4])<=20:
    from sklearn.externals import joblib
else:
    import joblib

def train_mapper(sample):

    img = sample
    if not os.path.exists(img):
        print(img, "图片不存在")

    proj, geotrans, img = GRID.read_img(img)
    img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]), order='C')
    return img

def train_r(train_list,size):
    dataLabel = []
    i = 0
    with open(train_list, "r") as f:
        lines = [line.strip() for line in f]
        dataNum = len(lines)
        dataMat = np.zeros((dataNum, size))
        for line in lines:
            img_path, lab = line.replace("\n", "").split("\t")
            dataLabel.append(int(lab))
            dataMat[i, :] = train_mapper(img_path)
            i = i + 1
    return dataMat, dataLabel

def test_mapper(sample):
    img = sample
    if not os.path.exists(img):
        print(img, "图片不存在")
    proj, geotrans, img = GRID.read_img(img)
    img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]), order='C')
    return img

def SVM(data_root_path,model_save_dir,row_patch_size,col_patch_size,channel,gamma,C):
    test_file_path = data_root_path + "test.txt"
    train_file_path = data_root_path + "train.txt"
    test_result_txt = model_save_dir + "/SVM_test_result_txt.txt"
    train_result_txt = model_save_dir + "/SVM_train_result_txt.txt"
    size=channel * row_patch_size * col_patch_size
    ###########################模型训练评估################################

    train_image, train_label = train_r(train_file_path,size)
    with open(test_result_txt, "w") as f:
        pass
    with open(train_result_txt, "w") as f:
        pass

    predictor = svm.SVC(gamma=gamma, C=C, decision_function_shape='ovr', kernel='rbf', probability=True)
    predictor.fit(train_image, train_label)
    joblib.dump(predictor, model_save_dir + '/SVM.model')
    result1 = []
    labels1 = []
    errorCount1 = 0.0
    with open(train_file_path, "r") as f:
        lines = [line.strip() for line in f]
        dataNum1 = len(lines)
        for line in lines:
            i = 0
            img_path, train_dataLabel = line.replace("\n", "").split("\t")
            train_dataMat = train_mapper(img_path)
            classifierResult1 = predictor.predict(train_dataMat)
            probability1 = predictor.predict_proba(train_dataMat)
            result1.append(probability1[0][1])
            labels1.append(int(train_dataLabel))
            with open(train_result_txt, "a") as f:
                line = "%s \t %d \t %f\n" % (img_path, int(train_dataLabel), probability1[0][1])
                f.write(line)
            if (classifierResult1[0] != int(train_dataLabel)):
                errorCount1 += 1.0
    errorCount = 0.0
    result = []
    labels = []
    pred=[]
    with open(test_file_path, "r") as f:
        lines = [line.strip() for line in f]
        dataNum = len(lines)
        for line in lines:
            img_path, test_dataLabel = line.replace("\n", "").split("\t")
            test_dataMat = test_mapper(img_path)
            classifierResult = predictor.predict(test_dataMat)
            probability = predictor.predict_proba(test_dataMat)
            pred.append(probability[0][1])
            labels.append(int(test_dataLabel))
            result.append(classifierResult[0])
            with open(test_result_txt, "a") as f:
                line = "%s \t %d \t %f\n" % (img_path, int(test_dataLabel), probability[0][1])
                f.write(line)
            if (classifierResult[0] != int(test_dataLabel)):
                errorCount += 1.0
        print("预测错的数据个数\t%d\n精度为%f" % (errorCount, (dataNum - errorCount) / dataNum))
        test_acc = (dataNum - errorCount) / dataNum
    print("训练集预测错的数据个数\t%d\n精度为%f" % (errorCount1, (dataNum1 - errorCount1) / dataNum1))
    train_acc = (dataNum1 - errorCount1) / dataNum1
    con_mat, accuracy, precision, recall, f1, RMSE, kappa, roc_auc, fpr, tpr = roc_s.evaluate_index(labels, result, pred)
    evaluate_txt = model_save_dir + '/evaluate_result.txt'
    with open(evaluate_txt, "w") as f:
        pass
    with open(evaluate_txt, "a") as f:
        line = "confusion_matrix: \n" + str(con_mat) + "\naccuracy:" + str(accuracy) + "\nprecision:" + str(
            precision) + "\nrecall:" + str(recall) + "\nf1:" + str(f1) + "\nroc_auc:" + str(roc_auc)
        f.write(line)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    lw = 3
    plt.figure(0)
    plt.plot(fpr, tpr, lw=lw, label='SVM:AUC=%0.4f' % roc_auc)
    plt.rcParams['savefig.dpi'] = 350
    plt.rcParams['figure.dpi'] = 350
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.tick_params(labelsize=18)
    plt.ylim([0.0, 1.0])
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.xlabel('FPR', fontsize=25,fontproperties='Times New Roman')
    plt.ylabel('TPR', fontsize=25,fontproperties='Times New Roman')
    plt.legend(loc="lower right", handlelength=4, fontsize=18)
    plt.savefig(model_save_dir+'/ROC.png', bbox_inches='tight')
    plt.close()
    return roc_auc,test_acc,train_acc

###################################################滑坡敏感性图#####################################################
def SVM_mapping(Layer_stacking_path,model_save,predict_dir,row_patch_size,col_patch_size,step):
    from time import time
    model_save_dir=model_save+'/SVM.model'
    start = time()
    proj, geotrans, data = GRID.read_img(Layer_stacking_path)
    predictor = joblib.load(model_save_dir)
    if int(row_patch_size) == int(step):
        import math
        img_use = np.zeros((data.shape[0], math.ceil(data.shape[1]/step)*step, math.ceil(data.shape[2]/step)*step))
        img_use[:, 0:data.shape[1], 0:data.shape[2]] = data
        img_new = np.zeros((img_use.shape[1], img_use.shape[2]))
        print(data.shape)
        channel, height, width = data.shape
        for i in range(height // row_patch_size):
            for j in range(width // col_patch_size):
                img = data[:, i * row_patch_size:(i + 1) * row_patch_size, j * col_patch_size:(j + 1) * col_patch_size]
                img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]), order='C')
                results = predictor.predict_proba(img)
                probability = results[0][1]
                img_new[i * row_patch_size:(i + 1) * row_patch_size,j * col_patch_size:(j + 1) * col_patch_size] = probability
            print("progress: %.2f %%" % ((float(i) / float(height // row_patch_size)) * 100.0))
        img_new=img_new[0:data.shape[1], 0:data.shape[2]]
    else:
        if (row_patch_size % 2) == 0:
            row_add_up = int(row_patch_size / 2 - 1)
            row_add_down = int(row_patch_size / 2 + 1)
        else:
            row_add_up = int(row_patch_size / 2 - 0.5)
            row_add_down = int(row_patch_size / 2 - 0.5)
        if (col_patch_size % 2) == 0:
            col_add_left = int(col_patch_size / 2 - 1)
            col_add_right = int(col_patch_size / 2 + 1)
        else:
            col_add_left = int(col_patch_size / 2 - 0.5)
            col_add_right = int(col_patch_size / 2 - 0.5)
        img_use = np.zeros((data.shape[0], data.shape[1] + row_add_up + row_add_down, data.shape[2] + col_add_left + col_add_right))
        img_use[:, row_add_up:data.shape[1] + row_add_up, col_add_left:data.shape[2] + col_add_left] = data
        img_new = np.zeros((data.shape[1], data.shape[2]))
        print(img_new.shape)
        channel, height, width = data.shape
        step=int(step)
        for i in range(int(height / int(step))):
            for j in range(int(width / int(step))):
                img = img_use[:, i:(i + row_patch_size), j:(j + col_patch_size)]
                img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]), order='C')
                results = predictor.predict_proba(img)
                probability = results[0][1]
                img_new[(i * step):(i * step + step), (j * step):(j * step + step)] = probability
            print("progress: %.2f %%" % ((float(i) / float(height)) * 100.0))
    GRID.write_img(predict_dir, proj, geotrans, img_new)
    end = time()
    time = end - start
    print("用时%fs" % time)
    print("SVM滑坡易发性制图完成！")


