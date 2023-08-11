# -*- coding: utf-8 -*-
'''
计算研究区各因子层之间的相关性（皮尔逊相关系数）
依赖于自己写的GRID库
输入：研究区各因子数据层的名称/一一对应的研究区各因子层的NoData值/研究区各因子层存放的根目录
输出：一个矩阵，每一行每一列分别表示一个因子层。计算后主对角线应全部为1.

位置：在其他所有程序运行之前，先对因子筛选。
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Common_use import GRID

# 删除裁剪后影像中的NoData值，保证相关性计算正确
def delect(img,NoData):
    if NoData == 2147483647:
        indices = np.where(img == NoData)
        img = np.delete(img, indices)
    elif NoData == -1:
        indices = np.where(img == NoData)
        img = np.delete(img, indices)
    else:
        indices = np.where(img < -50000)
        img = np.delete(img, indices)
    return img
def PCC(name,NoData,root_path,save_path,heatmap):
    data_all = []  # 存放所有因子层数据列表
    print('*****************************删除Nodata值，保证计算正确*****************************')
    for i in range(len(name)):
        img_path = root_path + "/" + name[i] + ".tif"
        proj, geotrans, data = GRID.read_img(img_path)  # 读数据
        data = data.reshape(data.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
        data = delect(data, NoData[i])  # 删除裁剪后影像中的NoData值，保证相关性计算正确
        print(data.shape)
        data_all.append(data)
    print('*****************************删除完成！*****************************')
    # 计算相关系数
    '''
    变量矩阵的一行表示一个随机变量；
    输出结果是一个相关系数矩阵, results[i][j]表示第i个随机变量与第j个随机变量的相关系数.
    np.corrcoef是求两条数据（或者是两个list）数据之间的相关系数（coefficient)
    所以就是求了这两列数的相关系数，结果为一个二维矩阵(2*2数组形式)的形式体现，对角线为1，反对角线则为该相关系数。
    "[0, 1]"这个代表第0行第一列的那个数值 即为 coefficient
    '''

    corrcoef1 = np.zeros((len(data_all), len(data_all)))  # 创建一个矩阵存放计算得到的相关系数
    print('*****************************开始计算皮尔逊相关系数*****************************')
    i = 0
    for m in data_all:
        j = 0
        for n in data_all:
            results = np.corrcoef(np.array(m), np.array(n))  # 计算相关系数
            if i == j:
                corrcoef1[i, j] = results[0, 0]
            else:
                corrcoef1[i, j] = results[0, 1]
            j = j + 1
        i = i + 1
    np.savetxt(save_path, np.c_[corrcoef1], fmt='%f', delimiter='\t')  # 保存成txt文件
    print("Correlation coefficient: \n", corrcoef1)
    if heatmap=='y':
        print('*****************************开始绘制热力图*****************************')
        # 绘制热力图
        plt.rcParams['savefig.dpi'] = 350  # 图片像素
        plt.rcParams['figure.dpi'] = 350  # 分辨率
        # 指定图形的字体
        fig, ax = plt.subplots(figsize=(len(data_all) + 1, len(data_all) + 1))
        sns.heatmap(corrcoef1, annot=True, vmax=1, vmin=-1, ax=ax, square=True, cmap="RdBu")
        ax.set_yticklabels(name, fontsize=10, rotation=360, horizontalalignment='right',
                           fontproperties='Times New Roman')
        ax.set_xticklabels(name, fontsize=10, rotation=45, horizontalalignment='center',
                           fontproperties='Times New Roman')
        plt.show()
    else:
        print('*****************************相关系数计算完成*****************************')

