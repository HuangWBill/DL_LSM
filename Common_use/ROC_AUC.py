# -*- coding: utf-8 -*-

'''
环境：paddlepaddle必须为2.0gpu版本
保存图片名称、真实类别、预测类别为txt文件
绘制ROC曲线，并标注AUC值，输出混淆矩阵和评价指标

位置：在其他所有程序运行之前，先对因子筛选。
1、混淆矩阵：  	 预测是非滑坡	          预测是滑坡
    实际是非滑坡	True Negative(TN)	False Positive(FP)
    实际是滑坡	False Negative(FN)	True Positive(TP)
2、总体精度（OA）准确率：预测正确的样本的占总样本的比例，取值范围为[0,1]，取值越大，模型预测能力越好。
   Acc=（TN+TP）/（TN+TP+FP+FN）
3、精确率：是分类器预测的正样本中预测正确的比例，取值范围为[0,1]，取值越大，模型预测能力越好。
   P=TP/（TP+FP）
4、召回率：分类器所预测正确的正样本占所有正样本的比例，取值范围为[0,1]，取值越大，模型预测能力越好。
   R=TP/（TP+FN）
5、F1值：精确率和召回率是一对矛盾的度量。一般来说，查准率高时，查全率往往偏低，而查全率高时，查准率往往偏低。所以通常只有在一些简单任务中，才可能使得查准率和查全率都很高。
需要综合考虑他们，最常见的方法就是在Precision和Recall的基础上提出了F1值的概念，来对Precision和Recall进行整体评价，F1是两者的加权调和平均。F1的定义如下：
   F1=（2*P*R）/（P+R）=（2*TP）/（2*TP+FP+FN）
6、均方根误差RMSE
7、kappa系数：这个系数的值越高，则代表模型实现的分类准确度越高。
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paddle
import paddle.fluid as fluid
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from Common_use import GRID


# 评价指标计算
def evaluate_index(labels, result,pred):
    con_mat = confusion_matrix(labels, result)  # 计算混淆矩阵
    accuracy = metrics.accuracy_score(labels, result)  # 计算准确率
    precision = metrics.precision_score(labels, result)  # 计算精确率
    recall = metrics.recall_score(labels, result)  # 计算召回率
    f1 = metrics.f1_score(labels, result)  # 计算F1值
    RMSE = (metrics.mean_squared_error(labels, result)) ** 0.5  # 计算均方根误差RMSE
    kappa = metrics.cohen_kappa_score(labels, result)  # 计算kappa系数
    fpr, tpr, threshold = roc_curve(labels, pred)  # 计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    return con_mat,accuracy,precision,recall,f1,RMSE,kappa,roc_auc,fpr, tpr

def ROC_single(save_model_dir,txt_path,save_path,model_type,ROC_type):
    paddle.enable_static()  # paddlepaddle2.0默认输入动态图，因此如果是静态图需要加这个语句;如果是1.8版本则不需要。
    # place = fluid.CPUPlace() #CPU训练
    place = fluid.CUDAPlace(0)  # GPU训练
    ## 构建测试用的执行器
    infer_exe = fluid.Executor(place)
    p = open(txt_path)
    line = p.readline()
    test_img = []
    label = []
    result = []
    labels = []
    pred=[]
    num = 0
    while line:
        img, label_img = line.split('\t', 1)
        test_img.append(img)
        label.append(label_img[0])
        line = p.readline()
        num = num + 1
    # 加载数据
    def load_img(path):
        proj, geotrans, img = GRID.read_img(path)  # 读数据
        return img

    # 加载LSTM数据
    def LSTM_load_img(path):
        proj, geotrans, img = GRID.read_img(path)  # 读数据
        k = 0
        # 读取图片内容
        hang = np.zeros((img.shape[0], img.shape[1] * img.shape[2]))
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                hang[:, k] = img[:, i, j]
                k = k + 1
        img = hang
        return img

    with open(save_path, "w") as f:
        pass
    if model_type == 'CNN3D':
        for i in range(num):
            infer_imgs = []  # 存放要预测图像数据
            infer_imgs.append(load_img(test_img[i]))  # 加载图片，并且将图片数据添加到待预测列表
            infer_imgs = np.array(infer_imgs).astype('float32')  # 转换成数组
            infer_imgs = np.expand_dims(infer_imgs, axis=0) # 增加一个维度，如（11，9，9）——（1，11，9，9）
            # 加载模型
            infer_program, feed_target_names, fetch_targets = \
                fluid.io.load_inference_model(save_model_dir, infer_exe)
            # 执行预测
            results = infer_exe.run(infer_program,  # 执行预测program
                                    feed={feed_target_names[0]: infer_imgs},  # 传入待预测图像数据
                                    fetch_list=fetch_targets)  # 返回结果

            rs = np.argmax(results[0])  # 取出预测结果中概率最大的元素索引值即预测结果
            pred.append(results[0][0][1])
            labels.append(int(label[i]))
            result.append(rs)
            line = "%s \t %d \t %d\n" % (test_img[i], int(label[i]), rs)  # 拼一行
            print(line)

            with open(save_path, "a") as f:  # 以追加模式打开存放结果文件
                line = "%s \t %d \t %f\t %d\n" % (test_img[i], int(label[i]), results[0][0][1], rs)  # 拼一行
                f.write(line)  # 写入文件
    elif model_type == 'LSTM':
        for i in range(num):
            infer_imgs = []  # 存放要预测图像数据
            infer_imgs.append(LSTM_load_img(test_img[i]))  # 加载图片，并且将图片数据添加到待预测列表
            infer_imgs = np.array(infer_imgs).astype('float32')  # 转换成数组
            # 加载模型
            infer_program, feed_target_names, fetch_targets = \
                fluid.io.load_inference_model(save_model_dir, infer_exe)
            # 执行预测
            results = infer_exe.run(infer_program,  # 执行预测program
                                    feed={feed_target_names[0]: infer_imgs},  # 传入待预测图像数据
                                    fetch_list=fetch_targets)  # 返回结果
            rs = np.argmax(results[0])  # 取出预测结果中概率最大的元素索引值即预测结果
            pred.append(results[0][0][1])
            labels.append(int(label[i]))
            result.append(rs)
            line = "%s \t %d \t %d\n" % (test_img[i], int(label[i]), rs)  # 拼一行
            print(line)

            with open(save_path, "a") as f:  # 以追加模式打开存放结果文件
                line = "%s \t %d \t %f\t %d\n" % (test_img[i], int(label[i]), results[0][0][1], rs)  # 拼一行
                f.write(line)  # 写入文件
    else:
        for i in range(num):
            infer_imgs = []  # 存放要预测图像数据
            infer_imgs.append(load_img(test_img[i]))  # 加载图片，并且将图片数据添加到待预测列表
            infer_imgs = np.array(infer_imgs)  # 转换成数组
            # 加载模型
            infer_program, feed_target_names, fetch_targets = \
                fluid.io.load_inference_model(save_model_dir, infer_exe)
            # 执行预测
            results = infer_exe.run(infer_program,  # 执行预测program
                                    feed={feed_target_names[0]: infer_imgs},  # 传入待预测图像数据
                                    fetch_list=fetch_targets)  # 返回结果
            rs = np.argmax(results[0])  # 取出预测结果中概率最大的元素索引值即预测结果
            pred.append(results[0][0][1])
            labels.append(int(label[i]))
            result.append(rs)
            line = "%s \t %d \t %d\n" % (test_img[i], int(label[i]), rs)  # 拼一行
            print(line)
            with open(save_path, "a") as f:  # 以追加模式打开存放结果文件
                line = "%s \t %d \t %f\t %d\n" % (test_img[i], int(label[i]), results[0][0][1], rs)  # 拼一行
                f.write(line)  # 写入文件

    # 评价指标计算
    con_mat,accuracy,precision,recall,f1,RMSE,kappa,roc_auc,fpr, tpr=evaluate_index(labels, result,pred)
    print('accuracy\tprecision\trecall\tf1\tRMSE\tkappa\troc_auc\n')
    print(accuracy,'\t',precision,'\t',recall,'\t',f1,'\t',RMSE,'\t',kappa,'\t',roc_auc,'\n')
    # 绘制ROC曲线图
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    lw = 3
    plt.figure(0)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC=%0.4f' % roc_auc)
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
    plt.savefig(save_model_dir+'/'+ROC_type+'_ROC.png', bbox_inches='tight')
    plt.close()


def read(txt):
    p = open(txt)
    line = p.readline()
    test_img=[]
    label=[]
    result=[]
    pred = []
    num=0
    while line:
        img, true_label, pre_pred, pre_label= line.split('\t', 3)
        test_img.append(img)
        label.append(int(true_label))
        pred.append(float(pre_pred))
        result.append(int(pre_label))
        line = p.readline()
        num=num+1
    return label,pred, result


def ROC_multi(txt,model_type,model_evaluate_result):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    # 清空txt文件
    with open(model_evaluate_result, "w") as f:
        pass

    con_mats, accuracys, precisions, recalls, f1s, RMSEs, kappas, roc_aucs=[],[],[],[],[],[],[],[]
    lw = 3
    plt.figure(0).clf()  # plt.close()将完全关闭图形窗口，其中plt.clf()将清除图形-您仍然可以在其上绘制另一个绘图。
    # SVM评价指标计算
    for i in range(len(model_type)):
        labels, pred, result = read(txt[i])
        con_mat, accuracy, precision, recall, f1, RMSE, kappa, roc_auc, fpr, tpr=evaluate_index(labels, result,pred)
        con_mats.append(con_mat)
        accuracys.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        RMSEs.append(RMSE)
        kappas.append(kappa)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=lw, label='%s:AUC=%0.4f' % (model_type[i],roc_auc))  # 假正率为横坐标，真正率为纵坐标做曲线
        with open(model_evaluate_result, "a") as f:  # 以追加模式打开存放结果文件
            line = str(model_type[i])+"_混淆矩阵:\n" + str(con_mat) + '\n'  # 拼一行
            f.write(line)  # 写入文件

    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 禁止Dateframe自动换行(设置为Flase不自动换行，True反之)
    pd.set_option('expand_frame_repr', False)
    # 保证输出结果对齐
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    evaluate_result = {}
    evaluate_result.update({'model': model_type})
    evaluate_result.update({'总体精度/准确率': accuracys})
    evaluate_result.update({'精确率': precisions})
    evaluate_result.update({'召回率': recalls})
    evaluate_result.update({'F1值': f1s})
    evaluate_result.update({'RMSE': RMSEs})
    evaluate_result.update({'kappa系数': kappas})
    evaluate_result.update({'AUC': roc_aucs})
    evaluate_result = pd.DataFrame(evaluate_result)

    with open(model_evaluate_result, "a") as f:  # 以追加模式打开存放结果文件
        f.write(str(evaluate_result))  # 写入文件
    # 绘制ROC曲线图
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 350  # 分辨率
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.tick_params(labelsize=18)
    plt.ylim([0.0, 1.0])
    plt.tick_params(labelsize=18)
    plt.xlabel('FPR', fontsize=25, fontproperties='Times New Roman')
    plt.ylabel('TPR', fontsize=25, fontproperties='Times New Roman')
    plt.legend(loc="lower right", handlelength=4, fontsize=18)
    plt.show()
    # 绘制模型评估结果柱状图
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 显示中文标签，且中文为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    evaluate_result.plot(kind='bar')
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.tick_params(labelsize=4)
    plt.legend(bbox_to_anchor=(1.025, 0), loc=3, borderaxespad=0, fontsize=4.5, borderpad=1.0, labelspacing=1.0)
    plt.show()
