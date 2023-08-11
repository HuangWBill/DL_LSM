# -*- coding: utf-8 -*-
'''
环境：paddlepaddle必须为1.8gpu版本
利用训练好的模型对整幅遥感影像进行预测并输出隶属度（概率）图/影像是多维的

输入：待预测影像存放位置（Layer_stacking.py生成的结果）/模型保存路径/预测后图片路径/训练影像行、列大小
输出：使用对应模型得到的滑坡易发性地图

位置："ROC-AUC-single_8.py" or "CNN_model_6.py" or "MLP_6.py"模型训练完成后进行；
'''

import numpy as np
import paddle
import paddle.fluid as fluid
import threading
from Common_use import GRID
from Common_use import SVM

# 多进程裁剪
def Block(img_path,num,row_patch_size,col_patch_size,save_path):
    from time import time
    start = time()
    proj, geotrans, data = GRID.read_img(img_path)  # 读原数据
    row = int(num ** 0.5)
    col = int(num / row)
    interval_row = int(data.shape[1] / row)
    interval_col = int(data.shape[2] / col)
    img_use=np.zeros((data.shape[0],data.shape[1]+2*row_patch_size,data.shape[2]+2*col_patch_size))
    img_use[:,row_patch_size:(data.shape[1]+row_patch_size),col_patch_size:(data.shape[2]+col_patch_size)]=data
    for i in range(row-1):
        for j in range(col-1):
            block_data = img_use[:, (interval_row*i):((i +1)* interval_row+2*row_patch_size),(interval_col*j):((j +1) * interval_col+2*col_patch_size)]
            GRID.write_img(save_path+ '/'+str(i)+'_'+str(j)+'.tif', proj, geotrans, block_data)  # 写数据
    for i in range(row-1):
        block_data = img_use[:, (interval_row * i):((i + 1) * interval_row + 2 * row_patch_size),(interval_col * (col-1)):]
        GRID.write_img(save_path+ '/' +str(i)+'_'+str(col-1)+'.tif', proj, geotrans, block_data)  # 写数据
    for j in range(col-1):
        block_data = img_use[:, (interval_row * (row-1)):,(interval_col*j):((j +1) * interval_col+2*col_patch_size)]
        GRID.write_img(save_path+ '/' +str(row-1)+'_'+str(j)+'.tif', proj, geotrans, block_data)  # 写数据
    block_data = img_use[:, (interval_row * (row-1)):,(interval_col * (col-1)):]
    GRID.write_img(save_path+ '/'+str(row-1)+'_'+str(col-1)+'.tif', proj, geotrans, block_data)  # 写数据
    print('分块完成！共%d块' % num)
    end = time()
    time = end - start
    print("分块用时%fs" % time)

# 多进程合并
def merge(num,save_path,row_patch_size,col_patch_size,model_type):
    from time import time
    start = time()
    row = int(num ** 0.5)
    col = int(num / row)
    datas=[]
    for i in range(row):
        path1 = save_path + '/pre'+str(i) + '_0.tif'
        proj, geotrans, A = GRID.read_img(path1)  # 读原数据
        A=A[row_patch_size:(A.shape[0]-row_patch_size),col_patch_size:(A.shape[1]-col_patch_size)]
        for j in range(1,col):
            path = save_path + '/pre' + str(i) + '_' + str(j) + '.tif'
            proj, geotrans, B = GRID.read_img(path)  # 读原数据
            B = B[row_patch_size:(B.shape[0] - row_patch_size), col_patch_size:(B.shape[1] - col_patch_size)]
            A = np.hstack((A, B))
        datas.append(A)
    merge_data=datas[0]
    for i in range(1,row):
        merge_data=np.vstack((merge_data,datas[i]))
    GRID.write_img(save_path+'/'+str(model_type)+'_LSM.tif', proj, geotrans, merge_data)  # 写数据
    print('拼接完成！大小为:', merge_data.shape)
    end = time()
    time = end - start
    print("拼接用时%fs" % time)


# 多进程调用
class TriangulationThread(threading.Thread):
    def __init__(self, threadID, func, args=()):
        super(TriangulationThread, self).__init__()
        self.threadID = threadID
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

def multi_predict(img_path, num, row_patch_size, col_patch_size, save_path,model_type,model_save_dir, step):
    # creating num threads
    Block(img_path, num, row_patch_size, col_patch_size, save_path)
    from time import time
    start = time()
    row = int(num ** 0.5)
    col = int(num / row)
    pro_name_list = []
    img_dirs = []
    class_img_dir=[]
    for i in range(row):
        for j in range(col):
            pro_name = 'Thread_' + str(i) + '_' + str(j)
            img_dir = save_path + '/' + str(i) + '_' + str(j) + '.tif'
            pre_img_dir = save_path + '/pre' + str(i) + '_' + str(j) + '.tif'
            pro_name_list.append(pro_name)
            img_dirs.append(img_dir)
            class_img_dir.append(pre_img_dir)
    print(pro_name_list)
    print(img_dirs)
    print(class_img_dir)
    for k in range(num):
        if model_type == 'SVM':
            pro_name_list[k] = TriangulationThread(pro_name_list[k], SVM.SVM_mapping, (img_dirs[k], model_save_dir, class_img_dir[k], row_patch_size, col_patch_size, step))
        else:
            pro_name_list[k] = TriangulationThread(pro_name_list[k], predict, (img_dirs[k], model_type, model_save_dir, class_img_dir[k], row_patch_size, col_patch_size, step))
        print(str(pro_name_list[k])+'  Start...')
        # 开启新线程
        pro_name_list[k].start()
    for l in range(len(pro_name_list)):
        # 等待所有线程完成
        pro_name_list[l].join()
    print('multi threads end!')
    end = time()
    time = end - start
    print("多线程预测用时%fs" % time)
    merge(num, save_path, row_patch_size,col_patch_size,model_type)


def predict(img_dir,model_type,model_save_dir,class_img_dir,row_patch_size,col_patch_size,step):
    from time import time
    paddle.enable_static()  # paddlepaddle2.0默认输入动态图，因此如果是静态图需要加这个语句;如果是1.8版本则不需要。
    start = time()
    place = fluid.CUDAPlace(0)  # GPU训练
    ## 构建测试用的执行器
    infer_exe = fluid.Executor(place)
    ## 指定作用域
    inference_scope = fluid.core.Scope()
    proj, geotrans, data = GRID.read_img(img_dir)  # 读数据
    print(data.shape)
    if int(row_patch_size) == int(step):
        import math
        img_use = np.zeros((data.shape[0], math.ceil(data.shape[1]/step)*step, math.ceil(data.shape[2]/step)*step))
        img_use[:, 0:data.shape[1], 0:data.shape[2]] = data
        img_new = np.zeros((img_use.shape[1], img_use.shape[2]))
        print(data.shape)
        channel, height, width = data.shape
        for i in range(height // row_patch_size):
            for j in range(width // col_patch_size):
                img = data[:, i * row_patch_size:(i + 1) * row_patch_size, j * col_patch_size:(j + 1) * col_patch_size]
                if model_type == 'LSTM':
                    k = 0
                    # 读取图片内容
                    hang = np.zeros((img.shape[0], img.shape[1] * img.shape[2]))
                    for m in range(img.shape[1]):
                        for n in range(img.shape[2]):
                            hang[:, k] = img[:, m, n]
                            k = k + 1
                    img = hang
                infer_imgs = []  # 存放要预测图像数据
                infer_imgs.append(img)  # 加载图片，并且将图片数据添加到待预测列表
                infer_imgs = np.array(infer_imgs).astype('float32')  # 转换成数组
                if model_type == 'CNN3D':
                    infer_imgs = np.expand_dims(infer_imgs, axis=0)  # 增加一个维度，如（11，9，9）——（1，11，9，9）
                # 加载模型
                with fluid.scope_guard(inference_scope):
                    infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(model_save_dir, infer_exe)
                    # 执行预测
                    results = infer_exe.run(infer_program,  # 执行预测program
                                            feed={feed_target_names[0]: infer_imgs},  # 传入待预测图像数据
                                            fetch_list=fetch_targets)  # 返回结果
                    probability = results[0][0][1]  # 取出预测结果中第一列的元素值，表示是滑坡的概率值
                img_new[i * row_patch_size:(i + 1) * row_patch_size,j * col_patch_size:(j + 1) * col_patch_size] = probability
            print("progress: %.2f %%" % ((float(i) / float(height // row_patch_size)) * 100.0))
        img_new=img_new[0:data.shape[1], 0:data.shape[2]]
    else:
        if (row_patch_size % 2) == 0:
            row_add_up = int(row_patch_size / 2 - 1)
            row_add_down = int(row_patch_size / 2 + 1)
        else:
            row_add_up = int(row_patch_size / 2 - 0.5)
            row_add_down = int(row_patch_size / 2 - 0.5)
        if (col_patch_size % 2) == 0:
            col_add_left = int(col_patch_size / 2 - 1)
            col_add_right = int(col_patch_size / 2 + 1)
        else:
            col_add_left = int(col_patch_size / 2 - 0.5)
            col_add_right = int(col_patch_size / 2 - 0.5)
        img_use = np.zeros((data.shape[0], data.shape[1] + row_add_up + row_add_down, data.shape[2] + col_add_left + col_add_right))
        img_use[:, row_add_up:data.shape[1] + row_add_up, col_add_left:data.shape[2] + col_add_left] = data
        img_new = np.zeros((data.shape[1], data.shape[2]))
        print(img_new.shape)
        channel, height, width = data.shape
        for i in range(int(height / int(step))):
            for j in range(int(width / int(step))):
                img = img_use[:, i:(i + row_patch_size), j:(j + col_patch_size)]
                if model_type == 'LSTM':
                    k = 0
                    # 读取图片内容
                    hang = np.zeros((img.shape[0], img.shape[1] * img.shape[2]))
                    for m in range(img.shape[1]):
                        for n in range(img.shape[2]):
                            hang[:, k] = img[:, m, n]
                            k = k + 1
                    img = hang
                infer_imgs = []  # 存放要预测图像数据
                infer_imgs.append(img)  # 加载图片，并且将图片数据添加到待预测列表
                infer_imgs = np.array(infer_imgs).astype('float32')  # 转换成数组
                if model_type == 'CNN3D':
                    infer_imgs = np.expand_dims(infer_imgs, axis=0)  # 增加一个维度，如（11，9，9）——（1，11，9，9）
                # 加载模型
                with fluid.scope_guard(inference_scope):
                    infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(model_save_dir,
                                                                                                    infer_exe)
                    # 执行预测
                    results = infer_exe.run(infer_program,  # 执行预测program
                                            feed={feed_target_names[0]: infer_imgs},  # 传入待预测图像数据
                                            fetch_list=fetch_targets)  # 返回结果
                    probability = results[0][0][1]  # 取出预测结果中第一列的元素值，表示是滑坡的概率值
                img_new[(i * step):(i * step + step), (j * step):(j * step + step)] = probability
            print("progress: %.2f %%" % ((float(i) / float(height)) * 100.0))
    GRID.write_img(class_img_dir, proj, geotrans, img_new)  # 写数据
    end = time()
    time = end - start
    print("用时%fs" % time)
    print("滑坡易发性制图完成！")


