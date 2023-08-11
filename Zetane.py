# -*- coding: utf-8 -*-
'''
Zetane 输入数据转换
tif ——————— npy
增加一个维度
'''

'''
import paddle
import paddle.fluid as fluid
import cv2
import numpy as np
from time import time
from wuqi import GRID
import os

paddle.enable_static()
img_dir = r"C:/Users\dell\Desktop" # 待预测影像存放位置

def file_filter(f):
    if f[-4:] in ['.tif']:
        return True
    else:
        return False

imgs = os.listdir(img_dir)  # 列出子目录中所有的文件
imgs = list(filter(file_filter, imgs))# 列出子目录中所有的.tif文件
print(imgs)
for i in range(len(imgs)):
    proj, geotrans, data = GRID.read_img(img_dir+'/'+imgs[i])  # 读数据
    print(img_dir+'/'+imgs[i])
    print(data.shape)
    #new_imgs=paddle.reshape(new_imgs, (-1, 12, 8,8))
    data = np.transpose(data, [2, 0, 1])
    new_imgs = np.expand_dims(data, axis=0)  # 增加一个维度，如（11，9，9）——（1，11，9，9）
    print(new_imgs.shape)

    print(img_dir+'/'+imgs[i][:-4]+'.npy')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    np.save(file=img_dir+'/'+imgs[i][:-4]+'.npy',arr=new_imgs)
'''




'''
Zetane 输出数据转换
npy ——————— tif

'''
import paddle
import paddle.fluid as fluid
import cv2
import numpy as np
from time import time
from wuqi import GRID
import os
'''
paddle.enable_static()
npy_dir = "C:/Users\dell\Desktop/as/qw.npy" # 待预测影像存放位置
data=np.load(npy_dir)
print(data.shape)
'''

'''
特征图可视化
'''
import paddle
import paddle.fluid as fluid
import cv2
import numpy as np
from time import time
from wuqi import GRID
import matplotlib.pyplot as plt
import os

def resize(img, size, resize_uniform=False, boder_color=[0, 0, 0]):
    H, W, C = img.shape
    if resize_uniform:
        scale_h = size / max(H, W)
        scale_w = scale_h
        dst_h = int(H * scale_h)
        dst_w = int(W * scale_h)
        img_resize = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        # 因为缩放的是图片的最大的方向，所以填充颜色是在小的图片方向上
        boder_size = max(abs(dst_h - H), abs(dst_w - W))
        # copyMakeBorder(img, top,bootom,left,right)
        img_resize = cv2.copyMakeBorder(img_resize, 0, 0, int(boder_size / 2), int(boder_size / 2), cv2.BORDER_CONSTANT, value=boder_color)
    else:
        img_resize = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        scale_h = size / H
        scale_w = size / W
    return img_resize, scale_h, scale_w

Base_map=r'C:/Users\dell\Desktop\0.tif'
feature_map=r'C:/Users\dell\Desktop\output0.npy'

proj, geotrans, data = GRID.read_img(Base_map)
data = np.transpose(data, [1, 2, 0])
print(data.shape)
data,a,b=resize(data,88)
print(data.shape)
#data = np.transpose(data, [2, 0,1])
print(data.shape)

plt.figure()
plt.imshow(data)

feature_data=np.load(feature_map)
feature_data=np.squeeze(feature_data,axis=0)
feature_data=np.squeeze(feature_data,axis=0)
print(feature_data.shape)

plt.imshow(feature_data, interpolation='bilinear', origin='lower',cmap='YlOrRd_r',alpha=0.5)
plt.show()
