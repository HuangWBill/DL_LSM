# -*- coding: utf-8 -*-
'''
将裁剪好的矩形各个因子层影像在深度方向叠加
依赖于自己写的GRID库
输入：各因子数据层的名称/研究区域各因子层的最小值和最大值/各裁剪后的矩形因子层所在文件夹目录/叠加后影像存放根目录
输出：各叠加好的滑坡与非滑坡区域的多维影像

注意：要求“矩形因子层所在文件夹目录”命名格式：
                            滑坡：landslide_dem/landslide_slp......
                            非滑坡：non_landslide_dem/non_landslide_slp......
        "各文件夹内部各影像“命名格式：
                            同一个滑坡或非滑坡的因子层名称必须严格一一对应，并用数字（1/2/3....）命名
                            同一个滑坡或非滑坡的因子层影像格式必须为”.tif“

'''

import math
import os
import random

import numpy as np

from Common_use import GRID


# 制作数据集
def dataset(name,min, max,root_path,new_img_path):
    # 将所有的滑坡影响因子存入列表
    for i in range(len(name)):
        path = root_path + "\\" + 'landslide_' + name[i]
        imgs = os.listdir(path)  # 列出子目录中所有的文件
        imgs = list(filter(GRID.file_filter, imgs))  # 列出子目录中所有的.tif文件
    # 将所有的非滑坡影响因子存入列表
    for i in range(len(name)):
        path = root_path + "\\" + 'non_landslide_' + name[i]
        non_imgs = os.listdir(path)  # 列出子目录中所有的文件
        non_imgs = list(filter(GRID.file_filter, non_imgs))  # 列出子目录中所有的.tif文件
    # 滑坡影像叠加
    for j in range(len(imgs)):
        b = []
        for i in range(len(name)):
            path = root_path + "\\" + 'landslide_' + name[i]
            img_path = path + "\\" + str(j) + ".tif"
            proj, geotrans, data = GRID.read_img(img_path)  # 读数据
            b.append(GRID.norm(data, min[i], max[i]))
        ##两幅图片在深度方向上叠加
        dst = b[0]
        for k in range(len(b) - 1):
            k = k + 1
            dst = np.dstack((dst, b[k]))
        dst = np.transpose(dst, [2, 0, 1])
        new_img = new_img_path + "/landslide" + "\\" + str(j) + '.tif'  # 拼完整叠加后影像路径
        if not os.path.exists(new_img_path + "/landslide"):
            os.makedirs(new_img_path + "/landslide")
        GRID.write_img(new_img, proj, geotrans, dst)  # 写数据
    print('*****************************滑坡影像叠加完成！*****************************')
    # 非滑坡影像叠加
    for j in range(len(non_imgs)):
        b = []
        for i in range(len(name)):
            path = root_path + "\\" + 'non_landslide_' + name[i]
            img_path = path + "\\" + str(j) + ".tif"
            proj, geotrans, data = GRID.read_img(img_path)  # 读数据
            b.append(GRID.norm(data, min[i], max[i]))
        ##两幅图片在深度方向上叠加
        dst = b[0]
        for k in range(len(b) - 1):
            k = k + 1
            dst = np.dstack((dst, b[k]))
        dst = np.transpose(dst, [2, 0, 1])
        new_img = new_img_path + "\\" + "non-landslide" + "\\" + "non-" + str(j) + '.tif'  # 拼完整叠加后影像路径
        if not os.path.exists(new_img_path + "/non-landslide"):
            os.makedirs(new_img_path + "/non-landslide")
        GRID.write_img(new_img, proj, geotrans, dst)  # 写数据
    print('*****************************非滑坡影像叠加完成！*****************************')


# 将图片路径存入name_data_list字典中
def save_train_test_file(name_data_list, path, name):
    if name not in name_data_list:  # 该类别不在字典中，则新建一个列表插入字典
        img_list = []
        img_list.append(path)  # 将图片路径存入列表
        name_data_list[name] = img_list  # 将图片列表插入字典
    else:  # 该类别在字典中，直接添加到列表
        name_data_list[name].append(path)

# 生成训练集和测试集
def data_txt(data_root_path,landslide_point_num,name_dict,rate):
    test_file_path = data_root_path + "test.txt"  # 测试文件路径
    train_file_path = data_root_path + "train.txt"  # 训练文件路径
    all_file_path = data_root_path + ".txt"  # 训练+测试文件路径
    name_data_list = {}  # 记录每个类别有哪些图片  key:有无滑坡  value:图片路径构成的列表

    def save_train_test_file(path, name):
        if name not in name_data_list:
            img_list = []
            img_list.append(path)
            name_data_list[name] = img_list
        else:
            name_data_list[name].append(path)
        return name_data_list

    def file_filter(f):
        if f[-4:] in ['.tif']:
            return True
        else:
            return False

    dirs = os.listdir(data_root_path) # 列出数据集目下所有的文件和子目录

    for d in dirs:
        full_path = os.path.join(data_root_path, d)
        if os.path.isdir(full_path):
            imgs = os.listdir(full_path)
            imgs = list(filter(file_filter, imgs))
            for img in imgs:
                name_data_list = save_train_test_file(os.path.join(full_path, img), d)
        else:
            pass

    # 将name_data_list字典中的内容写入文件
    ## 清空训练集和测试集文件
    with open(test_file_path, "w") as f:
        pass
    with open(train_file_path, "w") as f:
        pass
    with open(all_file_path, "w") as f:
        pass
    j = 0
    k = 0
    m = random.sample(range(1, landslide_point_num), math.ceil((landslide_point_num) * rate))  # random.sample()生成不相同的随机数
    # 遍历字典，将字典中的内容写入训练集和测试集
    for name, img_list in name_data_list.items():
        i = 1
        num = len(img_list)  # 获取每个类别图片数量
        print("%s: %d张" % (name, num))
        # 写训练集和测试集
        for img in img_list:
            if i in m:  # 每7笔写三笔测试集
                j = j + 1
                with open(test_file_path, "a") as f:  # 以追加模式打开测试集文件
                    line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                    f.write(line)  # 写入文件
                with open(all_file_path, "a") as f:  # 以追加模式打开测试集文件
                    line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                    f.write(line)  # 写入文件
            else:  # 训练集
                k = k + 1
                with open(train_file_path, "a") as f:  # 以追加模式打开测试集文件
                    line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                    f.write(line)  # 写入文件
                with open(all_file_path, "a") as f:  # 以追加模式打开测试集文件
                    line = "%s\t%d\n" % (img, name_dict[name])  # 拼一行
                    f.write(line)  # 写入文件
            i += 1  # 计数器加1
    print("训练集：%d张，测试集：%d张" % (k, j))
    print("*************************数据预处理完成！*****************************")


#训练集和测试集txt路径替换，满足增益比后文件
def new_data_txt(data_root_path,new_img_path):
    test_file_path = data_root_path + "test.txt"  # 原测试文件路径
    train_file_path = data_root_path + "train.txt"  # 原训练文件路径
    IGR_test_file_path = new_img_path + "test.txt"  # IGR后测试文件路径
    IGR_train_file_path = new_img_path + "train.txt"  # IGR后训练文件路径
    ## 清空训练集和测试集文件
    with open(IGR_train_file_path, "w") as f:
        pass
    with open(IGR_test_file_path, "w") as f:
        pass
    # 替换字符串
    with open(test_file_path, "r") as f1, open(IGR_test_file_path, "w") as f2:  # 以追加模式打开测试集文件
        for line in f1:
            line = line.replace("dataset", "IGR_dataset")
            f2.write(line)

    with open(train_file_path, "r") as f1, open(IGR_train_file_path, "w") as f2:  # 以追加模式打开测试集文件
        for line in f1:
            line = line.replace("dataset", "IGR_dataset")
            f2.write(line)
    print("训练集与测试集目录更新完成！")


# 滑坡相关因子层按照增益比计算后结果进行合成，用于预测
def Layer_stacking(name,min,max,root_path,new_img_path,h_min,h_max, l_min,l_max):
    # 滑坡影像叠加
    b = []
    for i in range(len(name)):
        img_path = root_path + "\\" + name[i] + ".tif"
        print(img_path)
        proj, geotrans, data = GRID.read_img(img_path)  # 读数据
        data = data[h_min:h_max, l_min:l_max]
        b.append(GRID.norm(data, min[i], max[i]))
    ##两幅图片在深度方向上叠加
    dst = b[0]
    for k in range(len(b) - 1):
        k = k + 1
        dst = np.dstack((dst, b[k]))
    dst = np.transpose(dst, [2, 0, 1])
    print(dst.shape)
    Layer_stacking_path = new_img_path + "/Factors_"+str(len(name))+"_Mapping.tif"  # 待预测影像存放位置
    GRID.write_img(Layer_stacking_path, proj, geotrans, dst)  # 写数据
    print("因子层叠加更新完成！")
    return Layer_stacking_path
