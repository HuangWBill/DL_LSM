# -*- coding: utf-8 -*-
'''
所有其他功能：
1、绘制热力图（相关系数）
计算研究区各因子层之间的相关性（皮尔逊相关系数）
依赖于自己写的GRID库
输入：研究区各因子数据层的名称/一一对应的研究区各因子层的NoData值/研究区各因子层存放的根目录
输出：一个矩阵，每一行每一列分别表示一个因子层。计算后主对角线应全部为1.

位置：在其他所有程序运行之前，先对因子筛选。
'''

import matplotlib.pyplot as plt
import os
import numpy as np
from Common_use import GRID
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import math
from pylab import mpl
from scipy.stats import norm


# 绘制训练测试的损失值和精度动态图
def dy_fig(train_batches,train_costs,train_accs,test_batches,test_costs,test_accs):
    plt.ion()  # 开启一个画图的窗口
    plt.pause(0.1)  # 暂停一秒
    plt.clf()  # 清除之前画的图
    # 第一行第一列图形
    ax1 = plt.subplot(2, 2, 1)
    # 第一行第二列图形
    ax2 = plt.subplot(2, 2, 2)
    # 第二行
    ax3 = plt.subplot(2, 2, 3)
    # 第二行
    ax4 = plt.subplot(2, 2, 4)
    # 选择ax1
    plt.sca(ax1)
    # 绘制训练损失值
    plt.plot(train_batches, train_costs)
    plt.title('train_cost')
    # 选择ax2
    plt.sca(ax2)
    # 绘制训练精度值
    plt.plot(train_batches, train_accs, color='r')
    plt.title('train_acc')
    # 选择ax3
    plt.sca(ax3)
    # 绘制测试损失值
    plt.plot(test_batches, test_costs, color='c')
    plt.title('test_cost')
    # 选择ax4
    plt.sca(ax4)
    # 绘制测试精度值
    plt.plot(test_batches, test_accs, color='b')
    plt.title('test_acc')

    plt.ioff()  # 关闭画图的窗口

# 绘制训练测试预测结果图（折线）
def train_test_fig(train_txt,test_txt):
    df_train = pd.read_csv(train_txt, sep=' ', header=None, names=['文件名', '真实标签', '预测结果', '预测标签'], index_col=False)
    df_test = pd.read_csv(test_txt, sep=' ', header=None, names=['文件名', '真实标签', '预测结果', '预测标签'], index_col=False)
    error_train = df_train['预测结果'] - df_train['真实标签']
    error_test = df_test['预测结果'] - df_test['真实标签']
    pre_df = [df_train['预测结果'], df_test['预测结果']]
    true_df = [df_train['真实标签'], df_test['真实标签']]
    pre = pd.concat(pre_df)
    true = pd.concat(true_df)
    MSE = mean_squared_error(true, pre)
    RMSE = sqrt(MSE)
    print(MSE)
    print(RMSE)
    # 求均值
    train_mean = np.mean(error_train)
    test_mean = np.mean(error_test)
    # 求方差
    train_var = np.var(error_train)
    test_var = np.var(error_test)
    # 求标准差
    train_std = np.std(error_train, ddof=1)
    test_std = np.std(error_test, ddof=1)

    landslide_point_num = df_train['真实标签'].value_counts()[1]
    landslide_point_num1 = df_test['真实标签'].value_counts()[1]
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    # plt.rcParams['figure.dpi'] = 300 # 分辨率
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()

    plt.subplot(311)
    # 训练集
    plt.plot(np.linspace(0, landslide_point_num, landslide_point_num), df_train.loc[0:landslide_point_num - 1, '预测结果'],
             lw=0.4, linestyle="-", c='blue', label='Outpus of train')
    plt.plot(np.linspace(0, landslide_point_num, landslide_point_num), df_train.loc[0:landslide_point_num - 1, '真实标签'],
             lw=0.8, linestyle="--", c='red', label='Targets of train')
    plt.plot(np.linspace(landslide_point_num, df_train.shape[0], df_train.shape[0] - landslide_point_num),
             df_train.loc[landslide_point_num:df_train.shape[0], '预测结果'], lw=0.4, linestyle="-", c='blue')
    plt.plot(np.linspace(landslide_point_num, df_train.shape[0], df_train.shape[0] - landslide_point_num),
             df_train.loc[landslide_point_num:df_train.shape[0], '真实标签'], lw=0.8, linestyle="--", c='red')
    # 测试集
    plt.plot(np.linspace(df_train.shape[0], df_train.shape[0] + landslide_point_num1, landslide_point_num1),
             df_test.loc[0:landslide_point_num1 - 1, '预测结果'], lw=0.4, linestyle="-", c='green', label='Outpus of test')
    plt.plot(np.linspace(df_train.shape[0], df_train.shape[0] + landslide_point_num1, landslide_point_num1),
             df_test.loc[0:landslide_point_num1 - 1, '真实标签'], lw=0.8, linestyle="--", c='orange',
             label='Targets of test')
    plt.plot(np.linspace(df_train.shape[0] + landslide_point_num1, df_train.shape[0] + df_test.shape[0],
                         df_test.shape[0] - landslide_point_num1),
             df_test.loc[landslide_point_num1:df_test.shape[0], '预测结果'], lw=0.4, linestyle="-", c='green')
    plt.plot(np.linspace(df_train.shape[0] + landslide_point_num1, df_train.shape[0] + df_test.shape[0],
                         df_test.shape[0] - landslide_point_num1),
             df_test.loc[landslide_point_num1:df_test.shape[0], '真实标签'], lw=0.8, linestyle="--", c='orange')
    plt.xlim(0, df_train.shape[0] + df_test.shape[0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Number of samples", fontproperties='Times New Roman')
    plt.ylabel("Targets and Outpus", fontproperties='Times New Roman')
    plt.legend(ncol=2)

    plt.subplot(312)
    # error
    # 训练集
    plt.plot(np.linspace(0, landslide_point_num, landslide_point_num), error_train.loc[0:landslide_point_num - 1],
             lw=0.4, linestyle="-", c='blue', label='Outpus of train')
    plt.plot(np.linspace(landslide_point_num, df_train.shape[0], df_train.shape[0] - landslide_point_num),
             error_train.loc[landslide_point_num:df_train.shape[0]], lw=0.4, linestyle="-", c='blue')
    # 测试集
    plt.plot(np.linspace(df_train.shape[0], df_train.shape[0] + landslide_point_num1, landslide_point_num1),
             error_test.loc[0:landslide_point_num1 - 1], lw=0.4, linestyle="-", c='green', label='Outpus of test')
    plt.plot(np.linspace(df_train.shape[0] + landslide_point_num1, df_train.shape[0] + df_test.shape[0],
                         df_test.shape[0] - landslide_point_num1),
             error_test.loc[landslide_point_num1:df_test.shape[0]], lw=0.4, linestyle="-", c='green')
    plt.xlim(0, df_train.shape[0] + df_test.shape[0])
    # plt.ylim(-0.1,1.1)
    plt.xlabel("Number of samples", fontproperties='Times New Roman')
    plt.ylabel("Errors", fontproperties='Times New Roman')
    plt.legend()

    # error分布
    # 训练集
    plt.subplot(325)
    n_train, bins_train, patches_train = plt.hist(error_train, 50)
    plt.xlabel("Errors", fontproperties='Times New Roman')
    plt.ylabel("Frequency", fontproperties='Times New Roman')
    # # 计算拟合线
    # y=1 / np.sqrt(2*np.pi) * np.exp(-(bins_train**2)/2)
    # #y = norm.pdf(bins_train, train_mean, train_std)
    # plt.plot(bins_train, y,lw=0.4,linestyle="-", c='red')

    # 测试集
    plt.subplot(326)
    n_test, bins_test, patches_test = plt.hist(error_test, 50)
    # # 计算拟合线
    # y = norm.pdf(bins_test, test_mean, test_std)
    # plt.plot(bins_test, y,lw=0.4,linestyle="-", c='red')

    # plt.xlim(0,df_train.shape[0]+df_test.shape[0])
    # plt.ylim(-0.1,1.1)
    plt.xlabel("Errors", fontproperties='Times New Roman')
    plt.ylabel("Frequency", fontproperties='Times New Roman')
    # plt.legend()
    #
    #
    plt.subplots_adjust(hspace=0.3)
    plt.show()

# train_txt='F:/rainfall-year\iIGR_conv_lstm_899/train_result_txt.txt'
# test_txt='F:/rainfall-year\iIGR_conv_lstm_899/test_result_txt.txt'
#train_test_fig(train_txt,test_txt)

# 根据信息增益比结果，绘制玫瑰图
def IGR_meigui_fig(IGR_result):
    IGR_df = pd.read_csv(IGR_result, sep=',', header=None, names=['因子名', 'IGR值'], index_col=False)
    # 降序排列
    IGR_df = IGR_df.sort_values(by='IGR值', axis=0, ascending=False)
    IGR_df = IGR_df.reset_index(drop=True)
    print(IGR_df)
    # 计算角度
    theta = np.linspace(0, 2 * np.pi, IGR_df.shape[0], endpoint=False)  # 360度等分成n份

    # 作图
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置画布
    fig = plt.figure(figsize=(12, 10))
    # 极坐标
    ax = plt.subplot(111, projection='polar')
    # 顺时针并设置N方向为0度
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')

    # 在极坐标中画柱形图
    ax.bar(theta, IGR_df['IGR值'], width=0.35, color=np.random.random((IGR_df.shape[0], 3)))

    ## 添加X轴的标签
    plt.xticks(theta, [])

    # 显示一些简单的图例
    for angle, data in zip(theta, IGR_df['IGR值']):
        if data > 0.09:
            ax.text(angle, data + 0.002, str(format(data, '.4f')), ha='center', va='center', fontsize=12)
        else:
            ax.text(angle, data + 0.02, str(format(data, '.4f')), ha='center', va='center', fontsize=12)
    ax.text(theta[0], 0.102, str('Rainfall'), ha='center', va='center', fontsize=12)
    ax.text(theta[1], 0.104, str('Altitude'), ha='center', va='center', fontsize=12)
    ax.text(theta[2], 0.108, str('Plan curvature'), ha='center', va='center', fontsize=12)
    ax.text(theta[3], 0.112, str('Profile curvature'), ha='center', va='center', fontsize=12)
    ax.text(theta[4], 0.105, str('NDVI'), ha='center', va='center', fontsize=12)
    ax.text(theta[5], 0.111, str('Distance to roads'), ha='center', va='center', fontsize=12)
    ax.text(theta[6], 0.104, str('TWI'), ha='center', va='center', fontsize=12)
    ax.text(theta[7], 0.104, str('Distance to faults'), ha='center', va='center', fontsize=12)
    ax.text(theta[8], 0.103, str('Lithology'), ha='center', va='center', fontsize=12)
    ax.text(theta[9], 0.104, str('Aspect'), ha='center', va='center', fontsize=12)
    ax.text(theta[10], 0.112, str('Distance to rivers'), ha='center', va='center', fontsize=12)
    ax.text(theta[11], 0.104, str('Slope'), ha='center', va='center', fontsize=12)
    ax.text(theta[12], 0.113, str('Surface roughness'), ha='center', va='center', fontsize=12)
    ax.text(theta[13], 0.110, str('Relief amplitude'), ha='center', va='center', fontsize=12)
    ax.text(theta[14], 0.104, str('Curvature'), ha='center', va='center', fontsize=12)

    ## 不显示Y轴
    ax.set_yticks([])
    plt.show()
# IGR_result='G:\python_projects\chuanzang_all_new/gr_result.txt'
#IGR_meigui_fig(IGR_result)

# 绘制滑坡点和影响因子关系图（折线）
# factor_txt_path='G:\python_projects\chuanzang_all_new/factor_analysis.txt'
#因子个数
# fn=15
# order = ['dem','slp','asp','cur','plancur','profilecur','faults','rivers','roads','lithology','Sroughness','qifudu','NDVI','TWI','rainfall']
# name = ['Altitude(m)','Slope(°)','Aspect','Curvature','Plan curvature','Profile curvature','Distance to faults(m)','Distance to rivers(m)','Distance to roads(m)',
#         'Lithology','Surface roughness','Relief amplitude','NDVI','TWI','ACR(mm/y)']
def factor_landslide_fig(factor_txt_path,fn,order,name):
    factor_df = pd.read_csv(factor_txt_path, sep=',')
    print(factor_df)
    # 坡向1-9替换为北、东北、东、东南、南、西南、西、西北、平地
    # asp_fenji=['北','东北','东','东南','南','西南','西','西北','平地']
    asp_fenji = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'F']
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 140  # 分辨率
    mpl.rcParams['font.size'] = 9
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    strings = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)"]
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(5, 3)
    for j in range(fn):
        #axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].hist(factor_df[order[j]], 60)
        axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].hist(factor_df[order[j]], 100, histtype='step')  # histtype='step',width=30
        if name[j] == 'Aspect':
            axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].set_xticks([1,2,3,4,5,6,7,8,9])
            axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].set_xticklabels(asp_fenji,fontsize=9,fontproperties='Times New Roman')
        if name[j] == 'Lithology':
            axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].set_xticks([1,2,3,4,5,6,7,8,9])
            axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].set_xticklabels([1,2,3,4,5,6,7,8,9],fontsize=9,fontproperties='Times New Roman')
        axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].text(0.85, 0.8, strings[j], fontsize=9, transform=axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].transAxes,)
        axs[math.floor(j / 3), j - 3 * math.floor(j / 3)].set_xlabel((name[j]), fontsize=9, fontproperties='Times New Roman')
        axs[math.floor(j / 3), 0].set_ylabel("Number of landslides", fontsize=9,fontproperties='Times New Roman')
    # plt.figure()
    # n, bins, patches=plt.hist(factor_df[order[0]], 100,width=30)#histtype='step',
    # plt.close()
    # plt.figure()
    # print(n, bins, patches)
    # patch = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    # plt.plot(patch, n)
    # plt.xlabel((name[0]), fontproperties='Times New Roman')
    # plt.ylabel("Frequency", fontproperties='Times New Roman')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
# factor_landslide_fig(factor_txt_path,fn,order,name)

'''
result_txt_path='E:/论文/English/chuanzang-English/rainfall_result.txt'
# 不同年份降雨结果绘制
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 300 # 分辨率
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
result_df = pd.read_csv(result_txt_path, sep=',',header=None, names=['Year', 'Susceptibility index'], index_col=False)
print(result_df)
plt.scatter(result_df['Susceptibility index'],result_df['Year'],c=result_df['Susceptibility index']*100,cmap='rainbow')
plt.yticks([1,1.2],labels=['2011','2016'],fontsize=9)
plt.vlines(x=0.542705, ymin=1.1, ymax=1.24, linestyles='dashed',colors='r')
plt.vlines(x=0.554591, ymin=0.96, ymax=1.1, linestyles='dashed',colors='r')
plt.ylabel("Year", fontproperties='Times New Roman',fontsize=10)
plt.xlabel("Susceptibility index", fontproperties='Times New Roman',fontsize=10)
plt.show()
'''

'''
#判断哪种岩性影响最大，进行排序
root_path = "E:\python_projects\chuanzang_landslide\chuanzang_process/factors\landslide_lithology" # 各裁剪后的矩形因子层所在文件夹目录
root_path1 = "E:\python_projects\chuanzang_landslide\chuanzang_process/factors/non_landslide_lithology" # 各裁剪后的矩形因子层所在文件夹目录

def tongji(path):
    imgs = os.listdir(path)  # 列出子目录中所有的文件
    imgs = list(filter(GRID.file_filter, imgs))  # 列出子目录中所有的.tif文件
    b = []
    a = []
    for j in range(len(imgs)):
        img_path = path + "\\" + str(j) + ".tif"
        proj, geotrans, data = GRID.read_img(img_path)  # 读数据
        c, d = np.unique(data, return_counts=True)
        m = len(c)
        for i in range(m):
            if c[i] in a:
                b[a.index(c[i])] += d[i]
            else:
                a.append(c[i])
                b.append(d[i])
    return a,b
landslide_value,landslide_count=tongji(root_path)
print('landslide_value:', landslide_value, '\n', 'landslide_count:', landslide_count)
non_landslide_value,non_landslide_count=tongji(root_path1)
print('non_landslide_value:', non_landslide_value, '\n', 'non_landslide_count:', non_landslide_count)
'''

# 根据PCC结果绘制热力图
# name = ['Altitude', 'Slope', 'Aspect', 'Curvature', 'Plan curvature', 'Profile curvature', 'Distance to faults','Distance to rivers', 'Distance to roads', 'lithology', 'Surface roughness', 'Relief amplitude', 'TWI','Rainfall', 'NDVI']
name = ['DEM', 'slp', 'asp', 'cur', 'plancur', 'profilecur', 'rivers', 'roads', 'lithology', 'SroughnessC', 'relief','rainfall', 'NDVI', 'TWI']
PCC_path='E:\论文\Remote sensing_SVM-LSM Toolbox\SVM_LSM_Toolbox_all\ArcGIS_Case\PCC.txt'
def heatmap(PCC_path,name):
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 100 # 分辨率
    A = np.zeros((len(name), len(name)), dtype=float)  # 先创建一个全零方阵A，并且数据的类型设置为float浮点型
    f = open(PCC_path)  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        print(list)
        A[A_row:] = list[0:len(name)]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    print(A)
    # 绘制热力图
    # 指定图形的字体
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(15, 15))
    ax=sns.heatmap(A, annot=True, annot_kws={'size':13},vmax=1,vmin = -1, ax=ax, square=True,cmap="RdBu_r",fmt='.3f', cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    # plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签，且中文为黑体
    # plt.rcParams['axes.unicode_minus']=False
    # ax.set_title('吴起县滑坡相关因子间相关系数', fontsize = 20)
    # ax.set_ylabel('因子层', fontsize = 15)
    # ax.set_xlabel('因子层', fontsize = 15)
    ax.set_yticklabels(name, fontsize=15, rotation=360, horizontalalignment='right', fontproperties='Times New Roman')
    ax.set_xticklabels(name, fontsize=15, rotation=30, horizontalalignment='center', fontproperties='Times New Roman')
    plt.show()
#heatmap(PCC_path,name)
'''
# 绘制易发性和某三个影响因子间的散点图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from wuqi import GRID
from osgeo import gdal
LSM_path='F:\python_projects\wuqi_landslide_CNN\wuqi_landslide_dataset\predict\裁剪_Landslide_Susceptibility Mapping_convLSTM.tif'
x_path='F:\python_projects\wuqi_landslide_CNN\wuqi_process\wuqi_factors\dem.tif'
y_path='F:\python_projects\wuqi_landslide_CNN\wuqi_process\wuqi_factors\dem.tif'
z_path='F:\python_projects\wuqi_landslide_CNN\wuqi_process\wuqi_factors\dem.tif'

proj, geotrans, x_data = GRID.read_img(x_path)  # 读数据
proj, geotrans, y_data = GRID.read_img(y_path)  # 读数据
proj, geotrans, z_data = GRID.read_img(z_path)  # 读数据
proj, geotrans, LSM_data = GRID.read_img(LSM_path)  # 读数据
x_data=x_data[2:14234,2:44414]
y_data=y_data[2:14234,2:44414]
z_data=z_data[2:14234,2:44414]
LSM_data=LSM_data[2:14234,2:44414]
x_data = x_data.reshape(x_data.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
y_data = y_data.reshape(y_data.size, order='C')
z_data = z_data.reshape(z_data.size, order='C')
LSM_data = LSM_data.reshape(LSM_data.size, order='C')
print(min(LSM_data))
print(max(LSM_data))
# 创建三维视图
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 设置三维数据，facecolors为第四维数据
cm = plt.cm.get_cmap('jet')
fig=ax.scatter3D(x_data, y_data, z_data, c = LSM_data, cmap=cm)
cb = plt.colorbar(fig,shrink=0.5, aspect=5)  #设置坐标轴
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cb.set_label('LSM')
plt.show()
'''

'''
# 绘制三维图
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#定义坐标轴
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

#生成三维数据
xx = [32,64]
yy = [500,600,650,700,750,800,900,1000]
X, Y = np.meshgrid(xx, yy)
print(X,Y)
Z = [[0.855697321,0.846657216],[0.858152531,0.842445348],[0.842085799,0.857443705],[0.823553583,0.846112755],
     [0.853653024,0.868548652],[0.853940664,0.838973126],[0.844582101,0.840236686],[0.854557035,0.822433843]]
X=np.array(X)
Y=np.array(Y)
Z=np.array(Z)
#作图
ax4.plot_surface(X,Y,Z)     #生成表面， alpha 用于控制透明度
# ax4.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面
# ax4.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
# ax4.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
#ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数
#设定显示范围
ax4.set_xlabel('batch_size')
#ax4.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax4.set_ylabel('buf_size')
#ax4.set_ylim(-4, 6)
ax4.set_zlabel('AUC')
#ax4.set_zlim(-3, 3)
plt.show()
'''

'''
# 绘制多个三维子图并加边框，适合5个参数的情况
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Common_use import GRID
from osgeo import gdal
result_path=r'G:/parameter_result_txt.txt'
result_df = pd.read_csv(result_path, sep='\t', index_col=False)
gamma=[3,7,9,11]
C=[3,7,9,11]
fig, axm = plt.subplots(figsize=(16, 10))
# axm.set(aspect=1)# xlim=(-15, 15), ylim=(-20, 5)
axm.spines['top'].set_visible(False)
axm.spines['right'].set_visible(False)
axm.spines['left'].set_visible(False)
axm.spines['bottom'].set_visible(False)
axm.set_xticks([])
axm.set_yticks([])
# # 创建三维视图
# fig = plt.figure()
plt.rcParams['savefig.dpi'] = 350
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# axins = zoomed_inset_axes(axm, zoom=0.5, loc='upper right')

xlabel=[' ']
ylabel=[' ']
xlabel=gamma+xlabel
ylabel=C+ylabel

inset_ax = fig.add_axes([0.03, 0.06, 0.88, 0.90], facecolor='white')  #
plt.xticks([1.5,5.8,10,14.2,16],labels=xlabel,fontsize=15)
plt.yticks([1.5,4,6.5,9,10],labels=ylabel,fontsize=15)
# inset_ax.set_xticks([1,6,10,14,16],labels=gamma,fontsize=9)
# inset_ax.set_yticks(C)


ax = []
for m in range(len(gamma)*len(C)):
    ax.append(m)
k=1
for i in range(len(gamma)):
    df = result_df[result_df['max_depth'] == gamma[i]]
    for j in range(len(C)):
        df1 = df[df['max_features'] == C[j]]
        # ax[k-1] = zoomed_inset_axes(axm, zoom=0.5, loc='upper right')
        ax[k-1] = fig.add_subplot(len(gamma),len(C),k,projection='3d')
        axc=ax[k-1].scatter3D(df1['n_estimators'], df1['min_samples_leaf'], df1['min_samples_split'], s=(df1['AUC']-min(df1['AUC']))/(max(df1['AUC'])-min(df1['AUC']))*100,c=df1['Train_acc-Test_acc'],cmap='rainbow',vmin=0, vmax=0.5)
        ax[k-1].set_xlabel('x')
        ax[k-1].set_ylabel('y')
        ax[k-1].set_zlabel('z')
        # plt.tight_layout()
        # ax[k].subplots_adjust(hspace=0, wspace=0)
        # ax[k].set_ylabel("min_samples_leaf", fontproperties='Times New Roman', fontsize=10)
        # ax[k].set_xlabel("max_depth", fontproperties='Times New Roman', fontsize=10)
        k=k+1
plt.subplots_adjust(left=0.01, top=0.95)
cax = plt.axes([0.93, 0.1, 0.025, 0.8])  # 左、下、宽度、 长度
cb=plt.colorbar(axc, cax=cax)
cb.ax.tick_params(labelsize=16)
# fig.colorbar(axc, ax=[ax[i-1] for i in range(len(gamma)*len(C))],fraction=0.02)#,fraction=0.02, pad=0.05
plt.show()
'''


