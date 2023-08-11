# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Common_use import GRID


def IGR(data_root_path,gr_result_txt,IGR_plt):
    train_file_path = data_root_path + ".txt"  # 训练文件路径
    # 生成标签向量
    def lab(train_dataLabel, lab_num):
        l = []
        if train_dataLabel == '0':
            for i in range(lab_num):
                l.append('否')
        elif train_dataLabel == '1':
            for i in range(lab_num):
                l.append('是')
        return l

    # 将矩阵转换为向量
    def rs(data, num):
        d = data[num, :, :]
        d = d.reshape(-1)
        return d

    label, dem, slp, asp, cur, plancur, profilecur, faults, rivers, roads, lithology, SroughnessC, qifudu, NDVI, TWI, rainfall = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    with open(train_file_path, "r") as f:
        lines = [line.strip() for line in f]  # 读取所有行，并去空格
        for line in lines:
            # 去掉一行数据的换行符，并按tab键拆分，存入两个变量
            img_path, train_dataLabel = line.replace("\n", "").split("\t")
            proj, geotrans, data = GRID.read_img(img_path)  # 读数据
            lab_num = data.shape[0] * data.shape[1] * data.shape[2]
            label.extend(lab(train_dataLabel, lab_num))
            for i in range(data.shape[0]):
                dem.extend(rs(data, 0))
                slp.extend(rs(data, 1))
                asp.extend(rs(data, 2))
                cur.extend(rs(data, 3))
                plancur.extend(rs(data, 4))
                profilecur.extend(rs(data, 5))
                faults.extend(rs(data, 6))
                rivers.extend(rs(data, 7))
                roads.extend(rs(data, 8))
                lithology.extend(rs(data, 9))
                SroughnessC.extend(rs(data, 10))
                qifudu.extend(rs(data, 11))
                NDVI.extend(rs(data, 12))
                TWI.extend(rs(data, 13))
                rainfall.extend(rs(data, 14))
    data = pd.DataFrame({'dem': dem, 'slp': slp, 'asp': asp, 'cur': cur, 'plancur': plancur, 'profilecur': profilecur, 'faults': faults,
                         'rivers': rivers, 'roads': roads,'lithology': lithology, 'SroughnessC': SroughnessC, 'qifudu': qifudu,
                         'NDVI': NDVI, 'TWI': TWI,'rainfall': rainfall, 'label': label})
    data[['dem', 'slp', 'asp', 'cur', 'plancur', 'profilecur', 'faults', 'rivers', 'roads', 'lithology', 'SroughnessC',
          'qifudu', 'NDVI', 'TWI', 'rainfall', 'label']]
    print(data)

    # 定义计算信息熵的函数：计算Infor(D)
    def infor(data):
        a = pd.value_counts(data) / len(data)
        return sum(np.log2(a) * a * (-1))

    # 定义计算信息增益的函数：计算g(D|A)
    def g(data, str1, str2):
        e1 = data.groupby(str1).apply(lambda x: infor(x[str2]))
        p1 = pd.value_counts(data[str1]) / len(data[str1])
        # 计算Infor(D|A)
        e2 = sum(e1 * p1)
        return infor(data[str2]) - e2

    # 定义计算信息增益率的函数：计算gr(D,A)
    def gr(data, str1, str2):
        return g(data, str1, str2) / infor(data[str1])

    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 禁止Dateframe自动换行(设置为Flase不自动换行，True反之)
    pd.set_option('expand_frame_repr', False)
    # 保证输出结果对齐
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    gr_result = pd.DataFrame({"IGR": [gr(data, 'dem', 'label'), gr(data, 'slp', 'label'), gr(data, 'asp', 'label'),
                                      gr(data, 'cur', 'label'), gr(data, 'plancur', 'label'),
                                      gr(data, 'profilecur', 'label'), gr(data, 'faults', 'label'),
                                      gr(data, 'rivers', 'label'), gr(data, 'roads', 'label'),
                                      gr(data, 'lithology', 'label'),
                                      gr(data, 'SroughnessC', 'label'), gr(data, 'qifudu', 'label'),
                                      gr(data, 'NDVI', 'label'), gr(data, 'TWI', 'label'),
                                      gr(data, 'rainfall', 'label')]},
                             index=['Altitude', 'Slope', 'Aspect', 'Curvature', 'Plan curvature', 'Profile curvature',
                                    'Distance to faults', 'Distance to rivers',
                                    'Distance to roads', 'Lithology', 'Surface roughness', 'Relief amplitude', 'NDVI',
                                    'TWI', 'Rainfall'])
    print('*******************************信息增益比结果*******************************')
    print(gr_result)
    # 清空txt文件
    with open(gr_result_txt, "w") as f:
        pass
    gr_result.to_csv(gr_result_txt, sep=',', header = False)
    if IGR_plt=='y':
        plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 显示中文标签，且中文为黑体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['savefig.dpi'] = 350  # 图片像素
        plt.rcParams['figure.dpi'] = 350  # 分辨率
        gr_result['pies'].plot(kind='bar')
        plt.xticks(rotation=15, horizontalalignment="center")
        plt.tick_params(labelsize=5)
        # plt.xlabel("滑坡相关因子",fontsize=5)
        # plt.ylabel("信息增益比",fontsize=5)
        plt.show()
    else:
        print('*******************************信息增益比计算完成！*******************************')
