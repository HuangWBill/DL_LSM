# -*- coding: utf-8 -*-
# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
from Common_use import ROC_AUC, predict_multi
from Common_use import SVM, CNN2D, LSTM
from Common_use import Conv_SE_LSTM as CAL
import os

txt_path = input('请输入参数txt文件路径：')
path = []
f = open(txt_path, encoding='utf-8')
list = f.readlines()
for line in list:
    name = line.split(' = ')
    path.append(name)


# 数据集存放根目录
new_dataset_path = eval(path[2][1])
# 用来预测的影像存放位置
predict_path = eval(path[4][1])

######################################模型训练，参数调优##########################################
# 分块行大小
row_size = eval(path[8][1])
# 分块列大小
col_size = eval(path[10][1])
# 通道数
channel = eval(path[12][1])
# 模型保存路径
model_save = eval(path[14][1])

#########DL参数###############
# 批次大小
batch_size = eval(path[24][1])
# 打乱的缓冲器大小
buf_size = eval(path[26][1])
# 学习率大小
learning_rate = eval(path[28][1])

print('1-模型训练，参数调优\n'
      '2-计算单个模型的评价指标(CNN2D、LSTM、Conv-SE-LSTM)\n'
      '3-最优参数预测生成滑坡易发性图(包括SVM的评价指标计算)\n'
      '4-计算所有最优模型的各项评价指标\n')
name_dict = {"landslide": 1, "non-landslide": 0}

type_size = 2
var = 1
while var > 0:
    step = input('请输入计算步骤类型(1-4)：')
    ######################################模型训练，参数调优##########################################
    if step == '1':
        model = eval(path[16][1])
        if model == 'SVM':
            import matplotlib.pyplot as plt
            import pandas as pd
            #########SVM参数###############
            gamma = eval(path[19][1])
            C = eval(path[20][1])

            SVM_result_txt = model_save + "/SVM/SVM_result_txt.txt"
            png_path = model_save + '/SVM/parameter_result_png.png'
            svg_path = model_save + '/SVM/parameter_result_png.svg'
            AUC = []
            test_acc, train_acc = [], []

            if not os.path.exists(model_save + "/SVM"):
                os.makedirs(model_save + "/SVM")
            with open(SVM_result_txt, "w") as f:
                pass
            with open(SVM_result_txt, "a") as f:
                line = "gamma\tC\tAUC\tTest_acc\tTrain_acc\tTrain_acc-Test_acc\n"  # 拼一行
                f.write(line)

            print("******************************开始训练!******************************")
            for i in gamma:
                for j in C:
                    model_save_dir = model_save + '/SVM/g_' + str(i) + '_C_' + str(j)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    roc_auc, testacc, trainacc = SVM.SVM(new_dataset_path, model_save_dir,row_size, col_size, channel, i, j)
                    with open(SVM_result_txt, "a") as f:  # 以追加模式打开存放结果文件
                        line = "%f\t%f\t%f\t%f\t%f\t%f\n" % (i, j, roc_auc, testacc, trainacc,trainacc-testacc)  # 拼一行
                        f.write(line)  # 写入文件
                    print("********************gamma=" + str(i) + "," + "C=" + str(j) + "SVM训练完成!********************")

            plt.rcParams['savefig.dpi'] = 350
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            result_df = pd.read_csv(SVM_result_txt, sep='\t', index_col=False)
            print(result_df)
            fig = plt.scatter(x=result_df['gamma'], y=result_df['C'], s=((result_df['AUC'] - min(result_df['AUC'])) / (max(result_df['AUC']) - min(result_df['AUC'])) * 100), c=result_df['Train_acc-Test_acc'],cmap='rainbow',vmin=0, vmax=0.5)
            cb = plt.colorbar(fig)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.ylabel("C", fontproperties='Times New Roman', fontsize=10)
            plt.xlabel("gamma", fontproperties='Times New Roman', fontsize=10)
            plt.savefig(png_path, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            print("*************************所有取值训练完成!*************************")

        if model == 'CNN2D':
            #########CNN2D参数###############
            CNN2D_conv_num1, CNN2D_drop1, CNN2D_conv_num2, CNN2D_drop2, CNN2D_fc, CNN2D_drop3 = eval(path[32][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/CNN2D/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    CNN2D.CNN(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel,BATCH_SIZE,
                              BUF_SIZE, CNN2D_conv_num1, CNN2D_drop1, CNN2D_conv_num2, CNN2D_drop2, CNN2D_fc,
                              CNN2D_drop3, learning_rate)

        if model == 'LSTM':
            #########LSTM参数###############
            LSTM_h_size, LSTM_h_layer, LSTM_fc_size, LSTM_drop_size = eval(path[36][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/LSTM/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    LSTM.LSTM(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, BATCH_SIZE,
                              BUF_SIZE, LSTM_h_size, LSTM_h_layer, LSTM_fc_size, LSTM_drop_size,learning_rate)

        else:
            #########Conv-SE-LSTM参数###############
            CAL_conv_num, CAL_h_size, CAL_h_layer, CAL_fc_size, CAL_drop_size = eval(path[40][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/Conv_SE_LSTM/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    CAL.Conv_SE_LSTM(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel,
                                     BATCH_SIZE,BUF_SIZE,learning_rate, CAL_conv_num, CAL_h_size, CAL_h_layer,
                                     CAL_fc_size,CAL_drop_size)

    ######################################单个模型评价##########################################
    if step == '2':
        model_type = eval(path[44][1])
        save_model_dir = eval(path[46][1])
        if model_type == 'SVM':
            print('SVM无需进行此步骤！')
        else:
            test_file_path = new_dataset_path + "test.txt"  # 测试文件路径
            train_file_path = new_dataset_path + "train.txt"  # 训练文件路径

            save_test_result_txt = save_model_dir + "/test_result_txt.txt"
            save_train_result_txt = save_model_dir + "/train_result_txt.txt"

            ROC_AUC.ROC_single(save_model_dir, train_file_path, save_train_result_txt, model_type, ROC_type='train')
            ROC_AUC.ROC_single(save_model_dir, test_file_path, save_test_result_txt, model_type, ROC_type='test')

    ######################################使用训练好的模型预测##########################################
    if step == '3':
        ######################################预测参数##########################################
        model_type = eval(path[44][1])
        save_model_dir = eval(path[46][1])
        step_size = eval(path[48][1])
        multi_select = eval(path[50][1])
        multi_num = eval(path[52][1])
        Layer_stacking_path = predict_path
        predict_path=os.path.dirname(predict_path)
        if multi_select == 'Y':
            predict_multi.multi_predict(Layer_stacking_path, multi_num, row_size, col_size, predict_path, model_type, save_model_dir, step_size)
        else:
            class_img_dir = predict_path + "/Landslide_Susceptibility_Mapping_" + str(model_type) + ".tif"
            if model_type == 'SVM':
                SVM_test_result_txt = save_model_dir + "/SVM_test_result_txt.txt"
                SVM_train_result_txt = save_model_dir + "/SVM_train_result_txt.txt"
                SVM.SVM_mapping(Layer_stacking_path, save_model_dir, class_img_dir, row_size, col_size, step_size)
            else:
                predict_multi.predict(Layer_stacking_path, model_type, save_model_dir, class_img_dir, row_size,
                                      col_size, step_size)

    ######################################所有的最优模型计算评价指标##########################################
    if step == '4':
        ######################################最优模型文件夹名称##########################################
        model_type=eval(path[46][1])
        txt = eval(path[58][1])
        model_evaluate_result = '/test_model_evaluate_result.txt'
        ROC_AUC.ROC_multi(txt,model_type,model_evaluate_result)

    if step == '0':
        print("退出程序！")
        break
