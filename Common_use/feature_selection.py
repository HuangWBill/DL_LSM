# -*- coding: utf-8 -*-

'''
计算PCC 和 IGR
'''


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from Common_use import GRID

def rs(data,num):
    d = data[num,:,:]
    d = d.reshape(-1)
    return d

def PCC(name,root_path,save_path,heatmap):
    print('*****************************开始计算皮尔逊相关系数*****************************')
    train_file_path = root_path + '.txt'
    PCC_path_txt = save_path + '\PCC.txt'
    name_list = []
    for k in range(len(name)):
        pro_name = name[k]
        name_list.append(pro_name)
        name_list[k] = []
    label = []

    with open(train_file_path, "r") as f:
        lines = [line.strip() for line in f]
        for line in lines:
            img_path, train_dataLabel = line.replace("\n", "").split("\t")
            proj, geotrans, data = GRID.read_img(img_path)
            lab_num = data.shape[1] * data.shape[2]
            label.extend(lab(train_dataLabel, lab_num))
            for i in range(data.shape[0]):
                name_list[i].extend(rs(data, i))
    del data, lab_num
    corrcoef1 = np.zeros((len(name), len(name)))
    for i in range(0, len(name)):
        for j in range(0, len(name)):
            results = np.corrcoef(name_list[i], name_list[j])
            if i == j:
                corrcoef1[i, j] = results[0, 0]
            else:
                corrcoef1[i, j] = results[0, 1]
    del results

    with open(PCC_path_txt, "w") as f:
        pass
    np.savetxt(PCC_path_txt, np.c_[corrcoef1], fmt='%f', delimiter='\t')
    if heatmap=='y':
        print('*****************************开始绘制热力图*****************************')
        # 绘制热力图
        plt.rcParams['savefig.dpi'] = 350  # 图片像素
        plt.rcParams['figure.dpi'] = 350  # 分辨率
        # 指定图形的字体
        plt.rc('font', family='Times New Roman')
        fig, ax = plt.subplots(figsize=(len(name), len(name)))
        sns.heatmap(corrcoef1, annot=True, annot_kws={'size': 12},vmax=1, vmin=-1, ax=ax, square=True, cmap="RdBu_r",
                    fmt='.3f', cbar=True, cbar_kws={"shrink": 0.8})
        plt.rcParams['axes.unicode_minus'] = False
        ax.set_yticklabels(name, fontsize=12, rotation=360, horizontalalignment='right')
        ax.set_xticklabels(name, fontsize=12, rotation=30, horizontalalignment='center')
        plt.savefig(save_path + '\PCC.png', bbox_inches='tight')
        plt.savefig(save_path + '\PCC.svg', bbox_inches='tight')
        #plt.show()
        print('*****************************相关系数计算完成*****************************')
    else:
        print('*****************************相关系数计算完成*****************************')

def lab(train_dataLabel,lab_num):
    l = []
    if train_dataLabel=='0':
        for i in range(lab_num):
            l.append('0')
    elif train_dataLabel=='1':
        for i in range(lab_num):
            l.append('1')
    return l

def ss(data,num):
    d = data[num,int(data.shape[1]/2):int(data.shape[1]/2)+1,int(data.shape[2]/2):int(data.shape[2]/2)+1]
    d = d.reshape(-1)
    return d

def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))

def g(data,str1,str2):
    e1 = data.groupby(str1).apply(lambda x:infor(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return infor(data[str2]) - e2

def gr(data,str1,str2):
    return g(data,str1,str2)/infor(data[str1])

def IGR(name,root_path,save_path,IGR_plt):
    print('*****************************开始计算信息增益比*****************************')
    train_file_path = root_path + '.txt'
    gr_result_txt = save_path + '\IGR.txt'

    name_list = []

    for k in range(len(name)):
        pro_name = name[k]
        name_list.append(pro_name)
        name_list[k] = []

    label = []

    with open(train_file_path, "r") as f:
        lines = [line.strip() for line in f]
        for line in lines:
            img_path, train_dataLabel = line.replace("\n", "").split("\t")
            proj, geotrans, data = GRID.read_img(img_path)
            label.extend(lab(train_dataLabel, 1))
            for i in range(data.shape[0]):
                name_list[i].extend(ss(data, i))
    del data

    data = {}
    for l in range(len(name)):
        data.update({name[l]: name_list[l]})
    data.update({'label': label})
    data = pd.DataFrame(data)

    name_new = list(name)
    name_new.append('label')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    gr_result = {}
    for l in range(len(name)):
        gr_result.update({name[l]: gr(data, name[l], 'label')})

    with open(gr_result_txt, "w") as f:
        pass
    with open(gr_result_txt, "a") as f:
        gr_result = pd.DataFrame(gr_result, index=[0]).T
        f.write(str(gr_result))
    if IGR_plt == 'y':
        print('*****************************开始绘制信息增益比图*****************************')
        plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 显示中文标签，且中文为黑体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['savefig.dpi'] = 350  # 图片像素
        plt.rcParams['figure.dpi'] = 350  # 分辨率
        gr_result.plot(kind='bar', legend=False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.xticks(rotation=15, horizontalalignment="center")
        plt.tick_params(labelsize=5)
        plt.savefig(save_path + '\IGR.png', bbox_inches='tight')
        plt.savefig(save_path + '\IGR.svg', bbox_inches='tight')
        #plt.show()
        print('*******************************信息增益比计算完成！*******************************')
    else:
        print('*******************************信息增益比计算完成！*******************************')


def VIF(name,root_path,save_path,IGR_plt):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print('*****************************开始计算方差膨胀因子和容忍度*****************************')
    train_file_path = root_path + '.txt'
    gr_result_txt = save_path + '\VIF.txt'

    name_list = []

    for k in range(len(name)):
        pro_name = name[k]
        name_list.append(pro_name)
        name_list[k] = []
    label = []

    with open(train_file_path, "r") as f:
        lines = [line.strip() for line in f]
        for line in lines:
            img_path, train_dataLabel = line.replace("\n", "").split("\t")
            proj, geotrans, data = GRID.read_img(img_path)
            label.extend(lab(train_dataLabel, 1))
            for i in range(data.shape[0]):
                name_list[i].extend(ss(data, i))
    del data

    data = {}
    for l in range(len(name)):
        data.update({name[l]: name_list[l]})
    data.update({'label': label})
    data = pd.DataFrame(data)

    data = data.drop(index=data[(data['label'] == '0')].index.tolist())
    data = data.drop('label', axis=1)
    data['const'] = 1

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    data = np.array(data)
    VIF=[variance_inflation_factor(data,l) for l in range(len(name))]
    TOL = [1/variance_inflation_factor(data, l) for l in range(len(name))]
    result = pd.DataFrame({'factors': name, "VIF": VIF,"TOL": TOL})
    #print(result)
    with open(gr_result_txt, "w") as f:
        pass
    with open(gr_result_txt, "a") as f:
        #results = pd.DataFrame(result, index=[0]).T
        f.write(str(result))
    if IGR_plt == 'y':
        print('*****************************开始绘制方差膨胀因子和容忍度图*****************************')
        result = result.set_index('factors')

        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.rcParams['figure.dpi'] = 250  # 分辨率
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False

        n = len(name)
        kinds = ['VIF','TOL']
        result = pd.concat([result, result.iloc[0:1,:]])  # 由于在雷达图中，要保证数据闭合，这里就再添加L列，并转换为np.ndarray
        centers = np.array(result.iloc[:, :])
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)  # 设置雷达图的角度,用于平分切开一个圆面
        angle = np.concatenate((angle, [angle[0]]))  # 为了使雷达图一圈封闭起来,需要下面的步骤

        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        #plt.subplots_adjust(hspace=0.17, wspace=0.22)
        ax = fig.add_subplot(gs[0,0], polar=True)  # 参数polar, 以极坐标的形式绘制图
        ## 添加X轴的标签
        plt.xticks(angle, [])
        ax.plot(angle, centers[:,0], linewidth=2, label=kinds[0])
        ax.set_thetagrids(angle * 180 / np.pi) #, result.index.values,fontsize=20
        ax.set_title('VIF',fontsize=30,x=0.5,y=1.1)
        for i in range(len(angle)):
            ax.text(angle[i], 26.2, str(result.index.values[i]), ha='center', va='center', fontsize=18)
        ax1 = fig.add_subplot(gs[0,1], polar=True)  # 参数polar, 以极坐标的形式绘制图
        plt.xticks(angle, [])
        ax1.plot(angle, centers[:, 1], linewidth=2, label=kinds[1])
        ax1.set_thetagrids(angle * 180 / np.pi)#, result.index.values,fontsize=20
        ax1.set_title('TOL',fontsize=30,x=0.5,y=1.1)
        for i in range(len(angle)):
            ax1.text(angle[i], 1.1, str(result.index.values[i]), ha='center', va='center', fontsize=18)
        plt.savefig(save_path + '\VIF.png', bbox_inches='tight')
        plt.savefig(save_path + '\VIF.svg', bbox_inches='tight')
        #plt.show()
        print('*******************************方差膨胀因子和容忍度计算完成！*******************************')
    else:
        print('*******************************方差膨胀因子和容忍度计算完成！*******************************')
