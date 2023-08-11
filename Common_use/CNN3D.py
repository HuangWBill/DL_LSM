# -*- coding: utf-8 -*-

'''
环境：paddlepaddle必须为1.8gpu版本
卷积神经网络CNN 训练模型，输出损失函数图及准确性图

输入：数据样本所在目录（3_Addfactors1_7.py生成的结果）/模型保存路径/训练影像行、列大小/通道数/分类结果个数
输出：CNN训练精度和损失值+CNN模型

注意：CNN内的结构和批次大小需要自己在里面修改，得到最优结果，适时停止迭代
     模型每一次迭代都会保存一个以数字（1,2,3......）命名的文件，需要自己选择

位置："Addfactors_2.py"+"Layer_stacking_4.py"+"data_pre-processing_3.py"运行完成后进行；后接"ROC-AUC-single_8.py"和"predict_multi_7.py"
'''

import os
from multiprocessing import cpu_count

import paddle
import paddle.fluid as fluid

from Common_use import GRID
from Common_use import others as ots

def parameter(Batch,Buf,conv_num1,conv_num2,fc_size,drop_size,learning_rate,parameter_txt):
    with open(parameter_txt, "w") as f:
        pass
    with open(parameter_txt, "a") as f:
        f.write('BATCH_SIZE = %d\nBUF_SIZE = %d\n\nfilter_size=3\nConv_filters_1=%d\nconv_stride = 1\nconv_padding = 1\n'
                'pool_size = 2\npool_stride = 2\npool_type = max_pool\n\n'
                'filter_size=3\nConv_filters_2=%d\nconv_stride = 1\nconv_padding = 1\n'
                'pool_size = 2\npool_stride = 2\npool_type = max_pool\n\n'
                'fc_num=%d\ndropout = %f\n\nlearning_rate=%f\n' % (Batch, Buf,conv_num1,conv_num2,fc_size,drop_size,learning_rate))

def train_mapper(sample):
    """
    根据传入的样本数据(一行文本)读取图片数据并返回
    :param sample: 元组，格式为(图片路径，类别)
    :return:返回图像数据、类别
    """
    img, label = sample # img为路基，label为类别
    if not os.path.exists(img):
        print(img, "图片不存在")

    # 读取图片内容
    proj, geotrans, img = GRID.read_img(img)  # 读数据
    k = 0
    # 读取图片内容
    '''
    hang = np.zeros((row_size * col_size, channel))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            hang[k, :] = img[:, i, j]
            k = k + 1
    '''
    return img, label  # 返回图像、类别

# 从训练集中读取数据
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f] # 读取所有行，并去空格
            for line in lines:
                # 去掉一行数据的换行符，并按tab键拆分，存入两个变量
                img_path, lab = line.replace("\n","").split("\t")
                yield img_path, int(lab) # 返回图片路径、类别(整数)
    return paddle.reader.xmap_readers(train_mapper, # 将reader读取的数进一步处理
                                      reader, # reader读取到的数据传递给train_mapper
                                      cpu_count(), # 线程数量
                                      buffered_size) # 缓冲区大小

def test_mapper(sample):  # sample估计就是reader返回的img，label
    img, label = sample
    if not os.path.exists(img):
        print(img, "图片不存在")
    proj, geotrans, img = GRID.read_img(img)  # 读数据
    k = 0
    '''
    # 读取图片内容
    hang = np.zeros((row_size * col_size, channel))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            hang[k, :] = img[:, i, j]
            k = k + 1
    '''
    return img, label

# 从测试集中读取数据
def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                # 去掉一行数据的换行符，并按tab键拆分，存入两个变量
                img_path, lab = line.replace("\n","").split("\t")
                yield img_path, int(lab) # 返回图片路径、类别(整数)
    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), buffered_size)

# 搭建CNN函数
# 结构：输入层 --> 卷积/激活/池化/dropout --> 卷积/激活/池化/dropout -->
#      卷积/激活/池化/dropout --> fc --> dropout --> fc(softmax)
def convolution_neural_network(image, type_size,conv_num1,conv_num2,fc_size,drop_size):
    """
    创建CNN
    :param image: 图像数据
    :param type_size: 输出类别数量
    :return: 分类概率
    paddle.fluid.layers.conv3d(input, # 形状为[N,C,D,H,W]或[N,D,H,W,C]的5-DTensor,N是批尺寸,C是通道数,D是特征深度,H是特征高度,W是特征宽度,数据类型为float16,float32或float64。
                               num_filters, # 滤波器（卷积核）的个数。和输出图像通道相同。
                               filter_size, # 滤波器大小。如果它是一个列表或元组，则必须包含三个整数值：（filter_size_depth, filter_size_height，filter_size_width）。若为一个整数，则filter_size_depth = filter_size_height = filter_size_width = filter_size。
                               stride=1, # 步长大小。滤波器和输入进行卷积计算时滑动的步长。如果它是一个列表或元组，则必须包含三个整型数：（stride_depth, stride_height, stride_width）。若为一个整数，stride_depth = stride_height = stride_width = stride。默认值：1。
                               padding=0, # 填充大小。默认值：0
                               dilation=1, # 膨胀比例大小。空洞卷积时会使用该参数，默认值=1。
                               groups=None, # 三维卷积层的组数。默认值=1。
                               param_attr=None,# 指定权重参数属性的对象。默认值为None，
                               bias_attr=None, # 指定偏置参数属性的对象。若 bias_attr 为bool类型，只支持为False，表示没有偏置参数。默认值为None。
                               use_cudnn=True, # 是否使用cudnn内核。只有已安装cudnn库时才有效。默认值：True。
                               act=None, # 激活函数类型，如tanh、softmax、sigmoid，relu等
                               name=None, # 一般无需设置，默认值：None
                               data_format="NCDHW" # 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCDHW"和"NDHWC"。默认值："NCDHW"。)
    paddle.fluid.layers.pool3d(input, # 形状为[N,C,D,H,W]或[N,D,H,W,C]的5-DTensor，N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度，数据类型为float32或float64。
                               pool_size=2, #  池化核的大小。如果它是一个元组或列表，那么它包含三个整数值，(pool_size_Depth, pool_size_Height, pool_size_Width)。若为一个整数，则表示D，H和W维度上均为该值，比如若pool_size=2, 则池化核大小为[2,2,2]。
                               pool_type='max', # 池化类型，可以为"max"或"avg"，"max" 对应max-pooling, "avg" 对应average-pooling。默认值："max"。
                               pool_stride=1, # 池化层的步长。如果它是一个元组或列表，那么它包含三个整数值，(pool_stride_Depth, pool_stride_Height, pool_stride_Width)。若为一个整数，则表示D，H和W维度上均为该值，比如若pool_stride=3, 则池化层步长为[3,3,3]。默认值：1。
                               pool_padding=0, #  池化填充。默认值：0
                               global_pooling=False, # 是否用全局池化。如果global_pooling = True，已设置的 pool_size 和 pool_padding 会被忽略， pool_size 将被设置为 [Din,Hin,Win] ， pool_padding 将被设置为0。默认值：False。
                               use_cudnn=True, # 是否使用cudnn内核。只有已安装cudnn库时才有效。默认值:True。
                               ceil_mode=False, # 是否用ceil函数计算输出的深度、高度和宽度。默认值：False。
                               name=None, # 一般无需设置。默认值：None。
                               exclusive=True, # 是否在平均池化模式忽略填充值。默认值：True。
                               data_format="NCDHW" # 输入和输出的数据格式，可以是"NCDHW"和"NDHWC"。默认值："NDCHW"。)
    """
    print(image.shape)
    # 第一组 卷积/激活/池化
    conv_1 = fluid.layers.conv3d(input=image, # 原始图像数据
                                 num_filters=conv_num1, # 卷积核数量
                                 filter_size=3, # 卷积核大小
                                 stride=1, # 卷积步长=1
                                 padding=1, # 卷积填充=1
                                 act="relu", # 激活函数
                                 data_format="NCDHW") # 输入的数据格式

    pool_1 = fluid.layers.pool3d(input=conv_1, # 第一层卷积结果图像数据
                                 pool_size=2, # 2*2区域池化
                                 pool_type='max', # 池化类型最大池化
                                 pool_stride=2, # 池化步长
                                 data_format="NCDHW") # 输入的数据格式
    conv1= conv_1.shape
    pool1=pool_1.shape

    # 第二组
    conv_2 = fluid.layers.conv3d(input=pool_1,  # 原始图像数据
                                 num_filters=conv_num2,  # 卷积核数量
                                 filter_size=3,  # 卷积核大小
                                 stride=1,  # 卷积步长=1
                                 padding=1,  # 卷积填充=1
                                 act="relu",  # 激活函数
                                 data_format="NCDHW")  # 输入的数据格式

    pool_2 = fluid.layers.pool3d(input=conv_2,  # 第一层卷积结果图像数据
                                 pool_size=2,  # 2*2区域池化
                                 pool_type='max',  # 池化类型最大池化
                                 pool_stride=2,  # 池化步长
                                 data_format="NCDHW")  # 输入的数据格式
    conv2 = conv_2.shape
    pool2 = pool_2.shape

    # 全连接层
    fc = fluid.layers.fc(input=pool_2, size=fc_size, act="relu")
    f=fc.shape
    # dropout
    drop = fluid.layers.dropout(x=fc, dropout_prob=drop_size)
    d1 = drop.shape
    # 输出层(fc)
    predict = fluid.layers.fc(input=drop, # 输入
                              size=type_size, # 输出值的个数
                              act="softmax") # 输出层采用softmax作为激活函数
    return predict,conv1,pool1,conv2,pool2,f,d1

def CNN(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,
        conv_num1,conv_num2,fc_size,drop_size,learning_rate):
    from time import time
    paddle.enable_static()  # paddlepaddle2.0默认输入动态图，因此如果是静态图需要加这个语句;如果是1.8版本则不需要。
    test_file_path = data_root_path + "test.txt"  # 测试文件路径
    train_file_path = data_root_path + "train.txt"  # 训练文件路径
    # 定义reader
    trainer_reader = train_r(train_list=train_file_path)  # 原始训练reader
    random_train_reader = paddle.reader.shuffle(reader=trainer_reader,
                                                buf_size=BUF_SIZE)  # 包装成随机读取器
    batch_train_reader = paddle.batch(random_train_reader,
                                      batch_size=BATCH_SIZE)  # 批量读取器

    test_reader = test_r(test_list=test_file_path)  # 原始测试reader
    random_test_reader = paddle.reader.shuffle(reader=test_reader,
                                               buf_size=BUF_SIZE)  # 包装成随机读取器
    batch_test_reader = paddle.batch(random_test_reader,
                                     batch_size=BATCH_SIZE)  # 批量读取器

    # 变量
    image = fluid.layers.data(name="image", shape=[1, channel, row_size, col_size], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # 调用函数，创建CNN
    predict, conv1, pool1, conv2, pool2, f, d1 = convolution_neural_network(image,type_size,conv_num1,conv_num2,fc_size,drop_size)
    print('conv1:', conv1, '\tpool1:', pool1, '\tconv2:', conv2, '\tpool2:', pool2, '\tf:', f, '\td1:', d1)
    # 损失函数:交叉熵
    cost = fluid.layers.cross_entropy(input=predict,  # 预测结果
                                      label=label)  # 真实结果
    avg_cost = fluid.layers.mean(cost)
    # 计算准确率
    accuracy = fluid.layers.accuracy(input=predict,  # 预测结果
                                     label=label)  # 真实结果
    # 从主程序中克隆一个测试程序
    test_program = fluid.default_main_program().clone(for_test=True)
    # 优化器
    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    optimizer.minimize(avg_cost)  # 将损失函数值优化到最小

    # 执行器
    # place = fluid.CPUPlace() #CPU训练
    place = fluid.CUDAPlace(0)  # GPU训练
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    # feeder
    feeder = fluid.DataFeeder(feed_list=[image, label],  # 指定要喂入数据
                              place=place)

    test_costs, test_accs = [], []
    train_costs, train_accs = [], []
    testing_costs, testing_accs = [], []
    training_costs, training_accs = [], []
    train_times = 0
    test_times = 0
    train_batches = []  # 迭代次数
    test_batches = []

    print('开始训练...')
    # 开始训练
    start = time()
    for pass_id in range(15000):
        train_times += 1
        for batch_id, data in enumerate(batch_train_reader()):  # 循环读取样本，执行训练
            train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                            feed=feeder.feed(data),  # 喂入参数
                                            fetch_list=[avg_cost, accuracy])  # 获取损失值、准确率
            training_accs.append(train_acc[0])  # 记录每次训练准确率
            training_costs.append(train_cost[0])  # 记录每次训练损失值
        train_batches.append(train_times)  # 记录迭代次数
        train_cost = sum(training_costs) / len(training_costs)  # 每轮的平均误差
        train_acc = sum(training_accs) / len(training_accs)  # 每轮的平均准确率
        train_accs.append(train_acc)
        train_costs.append(train_cost)

        test_times += 1
        for batch_id, data in enumerate(batch_test_reader()):
            test_cost, test_acc = exe.run(program=test_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost, accuracy])
            testing_costs.append(test_cost[0])
            testing_accs.append(test_acc[0])
        test_batches.append(test_times)
        test_cost = sum(testing_costs) / len(testing_costs)  # 每轮的平均误差
        test_acc = sum(testing_accs) / len(testing_accs)  # 每轮的平均准确率
        test_accs.append(test_acc)
        test_costs.append(test_cost)
        parameter_txt = model_save_dir + '/parameter_txt.txt'
        parameter(BATCH_SIZE, BUF_SIZE,conv_num1,conv_num2,fc_size,drop_size,learning_rate, parameter_txt)
        print("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f"
              % (pass_id, train_cost, train_acc, test_cost, test_acc))
        result_txt = model_save_dir + '/train_process.txt'
        with open(result_txt, "a") as f:
            f.write("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f\n" % (
                pass_id, train_cost, train_acc, test_cost, test_acc))
        # 训练过程可视化
        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)
        # 训练结束后，保存模型
        model_save_path = model_save_dir + '/CNN3D_' + str(pass_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        fluid.io.save_inference_model(dirname=model_save_path,
                                      feeded_var_names=["image"],
                                      target_vars=[predict],
                                      executor=exe)
    end = time()
    time = end - start
    print("用时%fs" % time)

    print('CNN3D:\t' + 'batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE) + '\tend!')





