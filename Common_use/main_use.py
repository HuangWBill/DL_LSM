# -*- coding: utf-8 -*-

from Common_use import Addfactors as add
from Common_use import Gain_ratio as igr
from Common_use import Pearson_cor_coeff as pcc
from Common_use import ROC_AUC, predict_multi
from Common_use import SVM, MLP, CNN2D, CNN3D, LSTM, Conv_Attention_LSTM

######################################每一个“=”后面必须加一个空格##########################################
######################################因子筛选##########################################
# 研究区所有因子数据层的名称
name = ['dem','slp','asp','cur','plancur','profilecur','faults', 'rivers','roads','lithology','SroughnessC','qifudu','NDVI','TWI','rainfall']
#按照name顺序输入所有因子层的最小值
min = [200, 0, -1, -19.923939, 0, 0, 0, 0, 0, 1, 1, 0, -0.9977095, -1.1832262, 296.6105]
#按照name顺序输入所有因子层的最大值
max = [8589, 89.451454, 359.99115, 23.270359, 81.51814, 58.136696, 47589.84, 126882.58, 115193.0, 9, 104.45102, 6275, 0.95180815, 28.639105, 2474.542]
# 按照name顺序输入所有因子层的NoData值
NoData = [-32768, -3.40282306074e+038, -3.40282306074e+038, -3.40282306074e+038, -3.40282306074e+038, -3.40282306074e+038,-3.40282306074e+038,-3.40282306074e+038,-3.40282306074e+038,-3.40282306074e+038,-3.40282306074e+038,-2147483648,-3.40282306074e+038,-3.40282306074e+038,-3.40282306074e+038]
# 研究区各因子层存放的根目录（研究区大范围）
factors_root_path = 'F:\python_projects\chuanzang_landslide\chuanzang_process\chuanzang_big_factors'
# 皮尔逊相关系数计算结果存放路径（如果研究区过大，自己电脑可能带不动！！！）
pcc_path = 'F:\python_projects\chuanzang_landslide\chuanzang_landslide_dataset\Pearson correlation coefficient.txt'
# 是否需要绘制热力图（y or n）
heatmap = 'y'
# 裁剪后的各数据的矩形因子层所在文件夹目录
factor_path = 'G:\python_projects\chuanzang_all_landslides\landslide_factors'
# 滑坡与非滑坡数据集影像存放根目录
old_dataset_path = 'G:\python_projects\chuanzang_comparison\changdu-bomi\dataset'
# 信息增益比计算结果存放目录
gr_result_txt = 'G:\python_projects\chuanzang_comparison\changdu-bomi\gr_result.txt'
# 滑坡点总数和测试集比例
landslide_point_num,rate = 115,0.3
# 是否需要绘制IGR图（y or n）
IGR_plt = 'y'

######################################筛选后更新数据##########################################
'''
No1
# 存放筛选后因子数据层的名称
new_name = ['rainfall','profilecur','plancur','NDVI','TWI', 'rivers','faults','asp','roads', 'lithology', 'slp','cur']
# 按照name顺序输入筛选后所有因子层的最小值
new_min = [296.6105, 0, 0, -0.9977095, -1.1832262, 0, 0, -1, 0, 1, 0,-19.923939]
# 按照name顺序输入筛选后所有因子层的最大值
new_max = [2474.542, 58.136696, 81.51814,0.95180815,28.639105, 126882.58,47589.84,359.99115, 115193.0,9,89.451454,23.270359]

No2
# 存放筛选后因子数据层的名称
new_name = ['cur', 'slp', 'lithology','roads','asp','faults', 'rivers','TWI','NDVI','plancur','profilecur','rainfall']
# 按照name顺序输入筛选后所有因子层的最小值
new_min = [-19.923939, 0, 1, 0, -1, 0, 0, -1.1832262, -0.9977095, 0, 0,296.6105]
# 按照name顺序输入筛选后所有因子层的最大值
new_max = [23.270359,89.451454,9, 115193.0,359.99115,47589.84, 126882.58,28.639105,0.95180815, 81.51814, 58.136696,2474.542]

No3
# 存放筛选后因子数据层的名称
new_name = ['slp','asp','cur','plancur','profilecur','TWI','lithology','faults','rainfall', 'rivers','roads','NDVI']
# 按照name顺序输入筛选后所有因子层的最小值
new_min = [0, -1, -19.923939, 0, 0, -1.1832262,0,0, 296.6105, 0, 0,-0.9977095]
# 按照name顺序输入筛选后所有因子层的最大值
new_max = [89.451454, 359.99115, 23.270359, 81.51814, 58.136696, 28.639105,9,47589.84, 2474.542, 126882.58, 115193.0,0.95180815]

No4
# 存放筛选后因子数据层的名称
new_name = ['slp','asp','cur','plancur','profilecur','faults', 'rivers','roads','lithology','NDVI','TWI','rainfall']
# 按照name顺序输入筛选后所有因子层的最小值
new_min = [0, -1, -19.923939, 0, 0, 0, 0, 0, 1, -0.9977095, -1.1832262, 296.6105]
# 按照name顺序输入筛选后所有因子层的最大值
new_max = [89.451454, 359.99115, 23.270359, 81.51814, 58.136696, 47589.84, 126882.58, 115193.0, 9, 0.95180815, 28.639105,2474.542]
'''

# 存放筛选后因子数据层的名称
new_name = ['slp','asp','cur','plancur','profilecur','faults', 'rivers','roads','lithology','NDVI','TWI','rainfall']
# 按照name顺序输入筛选后所有因子层的最小值
new_min = [0, -1, -19.923939, 0, 0, 0, 0, 0, 1, -0.9977095, -1.1832262, 296.6105]
# 按照name顺序输入筛选后所有因子层的最大值
new_max = [89.451454, 359.99115, 23.270359, 81.51814, 58.136696, 47589.84, 126882.58, 115193.0, 9, 0.95180815, 28.639105,2474.542]
# 因子层需要保留的行列最大最小值（h:行,l:列）
h_min,h_max, l_min,l_max = [1,15048,1,45890]
# 筛选后数据集存放根目录
new_dataset_path = 'G:\python_projects\chuanzang_all_landslides\landslide_dataset\LSTM\dataset_No4'
#new_dataset_path = 'E:\IGR\landslide_dataset\IGR_dataset'
# 筛选后用来预测的影像存放位置
predict_path = 'G:\python_projects\chuanzang_comparison/block\linzhi-lasa\predict'

######################################模型训练，参数调优##########################################
# 分块行大小
row_size = 8
# 分块列大小
col_size = 8
# 通道数
channel = 12
# 模型保存路径
model_save_dir = 'G:\python_projects\chuanzang_all_landslides\landslide_dataset\LSTM\model_No4'
#model_save_dir = 'E:\IGR\landslide_dataset\model'

#########SVM参数###############
gamma = [0.01, 0.02, 0.05, 0.1, 0.5, 1]
C = [0.01, 0.02, 0.05, 0.1, 0.5, 1]

#########MLP参数###############
# 批次大小
MLP_BATCH_SIZE = 64
# 打乱的缓冲器大小
MLP_BUF_SIZE = 600
# 隐藏层设置
MLP_h1_size,MLP_drop1,MLP_h2_size,MLP_drop2,MLP_h3_size,MLP_drop3 = 650,0.5,250,0.5,80,0.5
# 学习率大小
MLP_learning_rate = 0.0001

#########CNN2D参数###############
# 批次大小
CNN2D_BATCH_SIZE = 64
# 打乱的缓冲器大小
CNN2D_BUF_SIZE = 600
# 隐藏层设置
CNN2D_conv_num1,CNN2D_drop1,CNN2D_conv_num2,CNN2D_drop2,CNN2D_fc,CNN2D_drop3 = 64,0.5,132,0.5,350,0.5
# 学习率大小
CNN2D_learning_rate = 0.0001

#########CNN3D参数###############
# 批次大小
CNN3D_BATCH_SIZE = 30
# 打乱的缓冲器大小
CNN3D_BUF_SIZE = 1200
# 隐藏层设置
CNN3D_conv_num1,CNN3D_conv_num2,CNN3D_fc,CNN3D_drop = 12,24,150,0.5
# 学习率大小
CNN3D_learning_rate = 0.0001

#########LSTM参数###############
# 批次大小
LSTM_BATCH_SIZE = 32
# 打乱的缓冲器大小
LSTM_BUF_SIZE = 420
# 隐藏层设置
LSTM_h_size,LSTM_h_layer,LSTM_fc_size,LSTM_drop_size = 20,2,200,0.5
# 学习率大小
LSTM_learning_rate = 0.0001

#########Conv-LSTM参数###############
# 批次大小
convLSTM_BATCH_SIZE = 32
# 打乱的缓冲器大小
convLSTM_BUF_SIZE = 600
# 隐藏层设置，convLSTM_conv_num:分组卷积，每个通道使用的卷积层个数
convLSTM_conv_num,convLSTM_h_size,convLSTM_h_layer,convLSTM_fc_size,convLSTM_drop_size = 3,20,2,200,0.5
# 学习率大小
convLSTM_learning_rate = 0.0001

######################################最优模型文件夹名称##########################################
new_gamma = 0.01
new_C = 0.5
# 最优模型文件夹名称
model_best = ['mlp_426','ccnn_774','ccnn3D_417','iIGR_lstm_488','iIGR_conv_lstm_899']




name_dict = {"landslide":1, "non-landslide":0}
#分类结果的个数
type_size = 2
var = 1
while var > 0:
    print('1-进行因子筛选，包括皮尔逊相关系数计算（如果研究区过大，自己电脑可能带不动！！！），制作数据集，训练集和测试集划分，计算信息增益比\n'
          '2-根据筛选后的因子排序更新数据集，生成待预测的研究区影像\n'
          '3-模型训练，参数调优\n'
          '4-计算单个模型的评价指标\n'
          '5-最优参数预测生成滑坡易发性图\n'
          '6-计算所有最优模型的各项评价指标\n')
    step = input('请输入计算步骤类型(1-6)：')
    ######################################因子筛选##########################################
    if step == '1':
        # 计算皮尔逊相关系数
        pcc_if = input('是否计算皮尔逊相关系数(y or n)：')
        if pcc_if == 'y':
            pcc.PCC(name, NoData, factors_root_path, pcc_path, heatmap)
        else:
            print('不计算皮尔逊相关系数')

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

        # 计算信息增益比,目前仅接受如下顺序排列的15个因子
        # dem, slp, asp, cur, plancur, profilecur, faults, rivers, roads, lithology, SroughnessC, qifudu, NDVI, TWI, rainfall
        IGR_if = input('是否计算信息增益比(y or n)：')
        if IGR_if == 'y':
            igr.IGR(old_dataset_path, gr_result_txt, IGR_plt)
        else:
            print('不计算信息增益比')

    ######################################筛选后更新数据集##########################################
    if step == '2':
        Layer_stacking_path = predict_path + "/IGR_Factors_11_Mapping.tif"  # 待预测影像存放位置
        # 根据IGR值排序，更新数据集
        set_new = input('是否按照筛选后的更新数据集(y or n)：')
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
            add.Layer_stacking(new_name, new_min, new_max, factors_root_path, Layer_stacking_path, h_min, h_max, l_min,
                               l_max)
        else:
            print('不制作待预测的研究区整个影像')

    ######################################模型训练，参数调优##########################################
    if step == '3':
        model = input('请输入模型类型(SVM,MLP,CNN2D,CNN3D,LSTM,Conv-LSTM)：')
        if model == 'SVM':
            SVM_test_result_txt = model_save_dir + "/SVM_test_result_txt.txt"  # 生成txt保存原始标签及预测标签
            SVM_train_result_txt = model_save_dir + "/SVM_train_result_txt.txt"
            SVM_result_txt = model_save_dir + "/SVM_result_txt.txt"
            AUC = []
            test_acc, train_acc = [], []
            # 清空txt文件
            with open(SVM_result_txt, "w") as f:
                pass
            with open(SVM_result_txt, "a") as f:  # 以追加模式打开存放结果文件
                line = "gamma \t C \t AUC \t Test_acc \t Train_acc \n"  # 拼一行
                f.write(line)  # 写入文件

            print("******************************开始训练!******************************")
            for i in gamma:
                for j in C:
                    roc_auc, testacc, trainacc = SVM.SVM(new_dataset_path, SVM_test_result_txt, SVM_train_result_txt,
                                                         row_size, col_size, channel, i, j)
                    with open(SVM_result_txt, "a") as f:  # 以追加模式打开存放结果文件
                        line = "%f \t %f \t %f \t %f \t %f \t %f\n" % (i, j, roc_auc, testacc, trainacc,trainacc-testacc)  # 拼一行
                        f.write(line)  # 写入文件
                    print("********************gamma=" + str(i) + "," + "C=" + str(j) + "SVM训练完成!********************")
            print("*************************所有取值训练完成!*************************")

        if model == 'MLP':
            MLP.MLP(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, MLP_BATCH_SIZE,
                    MLP_BUF_SIZE,MLP_h1_size, MLP_drop1, MLP_h2_size, MLP_drop2, MLP_h3_size, MLP_drop3, MLP_learning_rate)

        if model == 'CNN2D':
            model_save_dir = model_save_dir + 'batch_' + str(CNN2D_BATCH_SIZE) + 'buf_' + str(CNN2D_BUF_SIZE)
            CNN2D.CNN(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, CNN2D_BATCH_SIZE,
                      CNN2D_BUF_SIZE, CNN2D_conv_num1, CNN2D_drop1, CNN2D_conv_num2, CNN2D_drop2, CNN2D_fc,
                      CNN2D_drop3, CNN2D_learning_rate)

        if model == 'CNN3D':
            CNN3D.CNN(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, CNN3D_BATCH_SIZE,
                      CNN3D_BUF_SIZE, CNN3D_conv_num1, CNN3D_conv_num2, CNN3D_fc, CNN3D_drop, CNN3D_learning_rate)

        if model == 'LSTM':
            model_save_dir = model_save_dir + 'batch_' + str(LSTM_BATCH_SIZE) + 'buf_' + str(LSTM_BUF_SIZE)
            LSTM.LSTM(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel, LSTM_BATCH_SIZE,
                      LSTM_BUF_SIZE, LSTM_h_size, LSTM_h_layer, LSTM_fc_size, LSTM_drop_size, LSTM_learning_rate)

        if model == 'Conv-LSTM':
            model_save_dir = model_save_dir + 'batch_' + str(convLSTM_BATCH_SIZE) + 'buf_' + str(convLSTM_BUF_SIZE)
            Conv_Attention_LSTM.convLSTM(type_size, new_dataset_path, model_save_dir, row_size, col_size, channel,
                                         convLSTM_BATCH_SIZE, convLSTM_BUF_SIZE, convLSTM_conv_num, convLSTM_h_size, convLSTM_h_layer, convLSTM_fc_size,
                                         convLSTM_drop_size, convLSTM_learning_rate)

    ######################################单个模型评价##########################################
    if step == '4':
        # 计算单个模型的AUC
        model_type = input('请输入模型类型(MLP,CNN2D,CNN3D,LSTM,Conv-LSTM)：')
        model_name = input('请输入模型文件夹名字(如IGR_conv_lstm_207)：')
        test_file_path = new_dataset_path + "test.txt"  # 测试文件路径
        train_file_path = new_dataset_path + "train.txt"  # 训练文件路径
        save_model_dir = model_save_dir + "/" + model_name  # 模型保存路径
        save_test_result_txt = save_model_dir + "/test_result_txt.txt"  # 生成txt保存原始标签及预测标签
        save_train_result_txt = save_model_dir + "/train_result_txt.txt"  # 生成txt保存原始标签及预测标签
        ROC_AUC.ROC_single(save_model_dir, train_file_path, save_train_result_txt, model_type)
        ROC_AUC.ROC_single(save_model_dir, test_file_path, save_test_result_txt, model_type)

    ######################################使用训练好的模型预测##########################################
    if step == '5':
        select = input('是否是SVM模型(y or n)：')
        Layer_stacking_path = predict_path + "/linzhi-lasa_Factors_12_Mapping.tif"  # 待预测影像存放位置
        if select == 'y':
            SVM_test_result_txt = model_save_dir + "/SVM_test_result_txt.txt"  # 生成txt保存原始标签及预测标签
            SVM_train_result_txt = model_save_dir + "/SVM_train_result_txt.txt"
            # SVM使用最优参数预测
            SVM.SVM_mapping(new_dataset_path, SVM_test_result_txt, SVM_train_result_txt, Layer_stacking_path, predict_path,row_size,
                            col_size, channel, new_gamma, new_C)
        else:
            # 使用训练好的模型预测(MLP,CNN2D,CNN3D,LSTM,Conv-LSTM)
            model_name = input('请输入模型文件夹名字(如IGR_conv_lstm_207)：')
            save_model_dir = model_save_dir + "/" + model_name  # 模型保存路径
            predict_multi.predict(Layer_stacking_path, predict_path, save_model_dir, row_size, col_size)

    ######################################所有的最优模型计算评价指标##########################################
    if step == '6':
        # 计算最优化模型的各项评价指标
        SVM_txt = model_save_dir + '\SVM_test_result_txt.txt'
        MLP_txt = model_save_dir + '/' + model_best[0] + '/test_result_txt.txt'
        CNN_txt = model_save_dir + '/' + model_best[1] + '/test_result_txt.txt'
        CNN3D_txt = model_save_dir + '/' + model_best[2] + '/test_result_txt.txt'
        LSTM_txt = model_save_dir + '/' + model_best[3] + '/test_result_txt.txt'
        Conv_LSTM_txt = model_save_dir + '/' + model_best[4] + '/test_result_txt.txt'
        model_evaluate_result = model_save_dir + '/test_model_evaluate_result.txt'  # 模型评估结果存放txt
        ROC_AUC_multi.ROC_multi(SVM_txt, MLP_txt, CNN_txt, CNN3D_txt, LSTM_txt, Conv_LSTM_txt, model_evaluate_result)

    if step == '0':
        print("退出程序！")
        break