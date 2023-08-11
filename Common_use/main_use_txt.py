# -*- coding: utf-8 -*-

from Common_use import Addfactors as add
from Common_use import feature_selection as fs
from Common_use import ROC_AUC, predict_multi
from Common_use import SVM, MLP, CNN2D, CNN3D, LSTM
from Common_use import Conv_Attention_LSTM as CAL
import os

# 读取txt并赋值
txt_path = input('请输入参数txt文件路径：')
path = []
f = open(txt_path, encoding='utf-8')
list = f.readlines()
for line in list:
    # 去掉一行数据的换行符，并按tab键拆分，存入两个变量
    name = line.split(' = ')
    path.append(name)

######################################因子筛选##########################################
# 研究区所有因子数据层的名称
name = eval(path[3][1])
# 按照name顺序输入所有因子层的最小值
min = eval(path[5][1])
# 按照name顺序输入所有因子层的最大值
max = eval(path[7][1])
# 按照name顺序输入所有因子层的NoData值
NoData = eval(path[9][1])
# 研究区各因子层存放的根目录（研究区大范围）
factors_root_path = eval(path[11][1])
# 特征选择计算结果存放目录（如果研究区过大，自己电脑可能带不动！！！）
PCC_IGR_path = eval(path[13][1])
# 是否需要绘制热力图（y or n）
heatmap = eval(path[15][1])
# 裁剪后的各数据的矩形因子层所在文件夹目录
factor_path = eval(path[17][1])
# 滑坡与非滑坡数据集影像存放根目录
old_dataset_path = eval(path[19][1])
# 滑坡点总数和测试集比例
landslide_point_num, rate = eval(path[21][1])
# 是否需要绘制IGR图或VIF图（y or n）
IGR_VIF_plt = eval(path[23][1])

######################################筛选后更新数据##########################################
# 存放筛选后因子数据层的名称
new_name = eval(path[27][1])
# 按照name顺序输入筛选后所有因子层的最小值
new_min = eval(path[29][1])
# 按照name顺序输入筛选后所有因子层的最大值
new_max = eval(path[31][1])
# 因子层需要保留的行列最大最小值（h:行,l:列）
h_min, h_max, l_min, l_max = eval(path[33][1])
# 筛选后数据集存放根目录
new_dataset_path = eval(path[35][1])
# 筛选后用来预测的影像存放位置
predict_path = eval(path[37][1])

######################################模型训练，参数调优##########################################
# 分块行大小
row_size = eval(path[41][1])
# 分块列大小
col_size = eval(path[43][1])
# 通道数
channel = eval(path[45][1])
# 模型保存路径
model_save = eval(path[47][1])

#########DL参数###############
# 批次大小
batch_size = eval(path[57][1])
# 打乱的缓冲器大小
buf_size = eval(path[59][1])
# 学习率大小
learning_rate = eval(path[61][1])

print('1-进行因子筛选，包括制作数据集，训练集和测试集划分，皮尔逊相关系数计算（如果研究区过大，自己电脑可能带不动！！！），计算信息增益比\n'
      '2-根据筛选后的因子排序更新数据集，生成待预测的研究区影像\n'
      '3-模型训练，参数调优\n'
      '4-计算单个模型的评价指标(MLP、CNN2D、CNN3D、LSTM、Conv-LSTM)\n'
      '5-最优参数预测生成滑坡易发性图(包括SVM的评价指标计算)\n'
      '6-计算所有最优模型的各项评价指标\n')
name_dict = {"landslide": 1, "non-landslide": 0}
# 分类结果的个数
type_size = 2
var = 1
while var > 0:
    step = input('请输入计算步骤类型(1-6)：')
    ######################################因子筛选##########################################
    if step == '1':
        # 制作数据集
        set_make = input('是否制作数据集(y or n)：')
        if set_make == 'y':
            add.dataset(name, min, max, factor_path, old_dataset_path)
        else:
            print('不制作数据集')

        # 训练集和测试集划分
        data_select = input('是否进行训练集和测试集的划分(y or n)：')
        if data_select == 'y':
            add.data_txt(old_dataset_path, landslide_point_num, name_dict, rate)
        else:
            print('不划分训练集和测试集')

        # 计算皮尔逊相关系数
        pcc_if = input('是否计算皮尔逊相关系数(y or n)：')
        if pcc_if == 'y':
            fs.PCC(name, old_dataset_path, PCC_IGR_path, heatmap)
        else:
            print('不计算皮尔逊相关系数')

        # 计算方差膨胀系数及容忍度
        pcc_if = input('是否计算方差膨胀系数及容忍度(y or n)：')
        if pcc_if == 'y':
            fs.VIF(name, old_dataset_path, PCC_IGR_path, IGR_VIF_plt)
        else:
            print('不计算方差膨胀系数及容忍度')

        # 计算信息增益比
        IGR_if = input('是否计算信息增益比(y or n)：')
        if IGR_if == 'y':
            fs.IGR(name, old_dataset_path, PCC_IGR_path, IGR_VIF_plt)
        else:
            print('不计算信息增益比')

    ######################################筛选后更新数据集##########################################
    if step == '2':
        # 根据IGR值排序，更新数据集
        set_new = input('是否按照特征选择后因子来更新数据集(y or n)：')
        if set_new == 'y':
            add.dataset(new_name, new_min, new_max, factor_path, new_dataset_path)
        else:
            print('不更新数据集')

        # 训练集和测试集txt路径替换，满足增益比后文件
        txt_new = input('是否更新训练集和测试集的划分(y or n)：')
        if txt_new == 'y':
            add.new_data_txt(old_dataset_path, new_dataset_path)
        else:
            print('不更新训练集和测试集的划分')

        # 滑坡相关因子层按照增益比计算后结果进行合成，用于预测
        img_new = input('是否根据筛选后的结果制作待预测的研究区整个影像(y or n)：')
        if img_new == 'y':
            Layer_stacking_path=add.Layer_stacking(new_name, new_min, new_max, factors_root_path, predict_path, h_min, h_max, l_min,l_max)
        else:
            print('不制作待预测的研究区整个影像')

    ######################################模型训练，参数调优##########################################
    if step == '3':
        model = eval(path[49][1])
        if model == 'SVM':
            import matplotlib.pyplot as plt
            import pandas as pd
            #########SVM参数###############
            gamma = eval(path[52][1])
            C = eval(path[53][1])

            SVM_result_txt = model_save + "/SVM/SVM_result_txt.txt"
            png_path = model_save + '/SVM/parameter_result_png.png'
            svg_path = model_save + '/SVM/parameter_result_png.svg'
            AUC = []
            test_acc, train_acc = [], []
            # 清空txt文件
            if not os.path.exists(model_save + "/SVM"):
                os.makedirs(model_save + "/SVM")
            with open(SVM_result_txt, "w") as f:
                pass
            with open(SVM_result_txt, "a") as f:  # 以追加模式打开存放结果文件
                line = "gamma\tC\tAUC\tTest_acc\tTrain_acc\tTrain_acc-Test_acc\n"  # 拼一行
                f.write(line)  # 写入文件

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

        if model == 'MLP':
            #########MLP参数###############
            # 隐藏层设置
            MLP_h1_size, MLP_drop1, MLP_h2_size, MLP_drop2, MLP_h3_size, MLP_drop3 = eval(path[65][1])
            # 学习率大小
            MLP_learning_rate = eval(path[63][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/MLP/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    # 3层MLP
                    MLP.MLP(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, BATCH_SIZE,
                            BUF_SIZE,MLP_h1_size, MLP_drop1, MLP_h2_size, MLP_drop2, MLP_h3_size, MLP_drop3, learning_rate)

        if model == 'CNN2D':
            #########CNN2D参数###############
            # 隐藏层设置
            CNN2D_conv_num1, CNN2D_drop1, CNN2D_conv_num2, CNN2D_drop2, CNN2D_fc, CNN2D_drop3 = eval(path[69][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/CNN2D/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    CNN2D.CNN(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel,BATCH_SIZE,
                              BUF_SIZE, CNN2D_conv_num1, CNN2D_drop1, CNN2D_conv_num2, CNN2D_drop2, CNN2D_fc,
                              CNN2D_drop3, learning_rate)

        if model == 'CNN3D':
            #########CNN3D参数###############
            # 隐藏层设置
            CNN3D_conv_num1, CNN3D_conv_num2, CNN3D_fc, CNN3D_drop = eval(path[73][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/CNN3D/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    CNN3D.CNN(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel,BATCH_SIZE,
                              BUF_SIZE, CNN3D_conv_num1, CNN3D_conv_num2, CNN3D_fc, CNN3D_drop,learning_rate)

        if model == 'LSTM':
            #########LSTM参数###############
            # 隐藏层设置
            LSTM_h_size, LSTM_h_layer, LSTM_fc_size, LSTM_drop_size = eval(path[77][1])
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/LSTM/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    LSTM.LSTM(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, BATCH_SIZE,
                              BUF_SIZE, LSTM_h_size, LSTM_h_layer, LSTM_fc_size, LSTM_drop_size,learning_rate)

        else:
            #########Conv-Attention-LSTM参数###############
            # 隐藏层设置，convLSTM_conv_num:分组卷积，每个通道使用的卷积层个数
            CAL_conv_num, CAL_h_size, CAL_h_layer, CAL_fc_size, CAL_drop_size = eval(path[81][1])
            # 注意力机制类型编号：不使用注意力机制——0，SE——1，CBAM——2，Non-local——3，CA——4
            attention_type = eval(path[83][1])
            attention=['','SE','CBAM','Non-local','CA']
            for BATCH_SIZE in batch_size:
                for BUF_SIZE in buf_size:
                    model_save_dir = model_save + '/Conv_'+str(attention[int(attention_type)])+'_LSTM/batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    CAL.Conv_Attention_LSTM(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel,
                                            BATCH_SIZE,BUF_SIZE,attention_type,learning_rate, CAL_conv_num, CAL_h_size, CAL_h_layer,
                                            CAL_fc_size,CAL_drop_size)

    ######################################单个模型评价##########################################
    if step == '4':
        # 计算单个模型的AUC
        model_type = eval(path[87][1])
        save_model_dir = eval(path[89][1])
        if model_type == 'SVM':
            print('SVM无需进行此步骤！')
        else:
            test_file_path = new_dataset_path + "test.txt"  # 测试文件路径
            train_file_path = new_dataset_path + "train.txt"  # 训练文件路径

            save_test_result_txt = save_model_dir + "/test_result_txt.txt"  # 生成txt保存原始标签及预测标签
            save_train_result_txt = save_model_dir + "/train_result_txt.txt"  # 生成txt保存原始标签及预测标签

            ROC_AUC.ROC_single(save_model_dir, train_file_path, save_train_result_txt, model_type, ROC_type='train')
            ROC_AUC.ROC_single(save_model_dir, test_file_path, save_test_result_txt, model_type, ROC_type='test')

    ######################################使用训练好的模型预测##########################################
    if step == '5':
        ######################################预测参数##########################################
        model_type = eval(path[87][1])
        save_model_dir = eval(path[89][1])
        # 步长大小
        step_size = eval(path[91][1])
        # 是否多线程预测
        multi_select = eval(path[93][1])
        # 多线程个数
        multi_num = eval(path[95][1])
        Layer_stacking_path = predict_path + "/Factors_"+str(len(new_name))+"_Mapping.tif"   # 待预测影像存放位置
        if multi_select == 'Y':
            predict_multi.multi_predict(Layer_stacking_path, multi_num, row_size, col_size, predict_path, model_type, save_model_dir, step_size)
        else:
            class_img_dir = predict_path + "/Landslide_Susceptibility_Mapping_" + str(model_type) + ".tif"  # 预测后图片路径
            if model_type == 'SVM':
                SVM_test_result_txt = save_model_dir + "/SVM_test_result_txt.txt"  # 生成txt保存原始标签及预测标签
                SVM_train_result_txt = save_model_dir + "/SVM_train_result_txt.txt"
                # SVM使用最优参数预测
                SVM.SVM_mapping(Layer_stacking_path, save_model_dir, class_img_dir, row_size, col_size, step_size)
            else:
                # 使用训练好的模型预测(MLP,CNN2D,CNN3D,LSTM,Conv-LSTM)
                predict_multi.predict(Layer_stacking_path, model_type, save_model_dir, class_img_dir, row_size,
                                      col_size, step_size)

    ######################################所有的最优模型计算评价指标##########################################
    if step == '6':
        ######################################最优模型文件夹名称##########################################
        # 计算最优化模型的各项评价指标
        model_type=eval(path[99][1])
        txt = eval(path[101][1])
        model_evaluate_result = '/test_model_evaluate_result.txt'  # 模型评估结果存放txt
        ROC_AUC.ROC_multi(txt,model_type,model_evaluate_result)

    if step == '0':
        print("退出程序！")
        break
