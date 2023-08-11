# -*- coding: utf-8 -*-

'''
环境：paddlepaddle必须为1.8gpu版本
卷积神经网络CNN 训练模型，输出损失函数图及准确性图

输入：数据样本所在目录（Addfactors1_7.py生成的结果）/模型保存路径/训练影像行、列大小/通道数/分类结果个数
输出：CNN训练精度和损失值+CNN模型

注意：CNN内的结构和批次大小需要自己在里面修改，得到最优结果，适时停止迭代
     模型每一次迭代都会保存一个以数字（1,2,3......）命名的文件，需要自己选择

位置："Addfactors.py"+"Layer_stacking_4.py"+"data_pre-processing_3.py"运行完成后进行；后接"ROC-AUC-single_8.py"和"predict_multi_7.py"
'''

import os
from multiprocessing import cpu_count

import paddle
import paddle.fluid as fluid

from Common_use import GRID
from Common_use import others as ots

def parameter(Batch,Buf,conv_num1,drop1,conv_num2,drop2,fc,drop3,learning_rate,parameter_txt):
    with open(parameter_txt, "w") as f:
        pass
    with open(parameter_txt, "a") as f:
        f.write('BATCH_SIZE = %d\nBUF_SIZE = %d\n\nfilter_size=3\nConv_filters_1=%d\nconv_stride = 1\nconv_padding = 1\n'
                'pool_size = 2\npool_stride = 2\npool_type = max_pool\ndropout1 = %f\n\n'
                'filter_size=3\nConv_filters_2=%d\nconv_stride = 1\nconv_padding = 1\n'
                'pool_size = 2\npool_stride = 2\npool_type = max_pool\ndropout2 = %f\n\n'
                'fc_num=%d\ndropout3 = %f\n\nlearning_rate=%f\n' % (Batch,Buf,conv_num1,drop1,conv_num2,drop2,fc,drop3,learning_rate))



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
def convolution_neural_network(image, type_size,conv_num1,drop1,conv_num2,drop2,fc,drop3):
    """
    创建CNN
    :param image: 图像数据
    :param type_size: 输出类别数量
    :return: 分类概率
    paddle.fluid.nets.simple_img_conv_pool(
                                    input,# 原始图像数据
                                    num_filters, filter_size, # 卷积核数量、卷积核大小
                                    pool_size, pool_stride,pool_padding=0, # 池化层大小、池化步长、池化层填充0（可选）
                                    pool_type='max',  # 池化类型有max和avg，默认max池化（可选）
                                    global_pooling=False, # 是否使用全局池化。如果为true，则忽略pool_size和pool_padding。默认为False
                                    conv_stride=1, conv_padding=0, # 卷积层步长，默认为1（可选）、卷积层填充，默认为0（可选）
                                    conv_dilation=1, conv_groups=1,# 卷积核膨胀，默认为1（可选）、卷积层组数，默认为1（可选）
                                    param_attr=None,# 权重参数(ParamAttr|None，可选)，默认值:None
                                    bias_attr=None, # 偏置参数(ParamAttr|None，可选)，如果设置为False，则不添加bias。如果设置为None，则将其初始化为零。默认值：None
                                    act=None,# 激活函数（’relu','softmax','sigmoid'，可选），如果设置为None，则不附加激活。默认值：None。
                                    use_cudnn=True #是否使用cudnn内核，仅在安装cudnn库时才有效。默认值：True。
name (str|None，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 Name ，默认值为None)
    """

    # 第一组 卷积/激活/池化/dropout
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image, # 原始图像数据
                                                  filter_size=3, # 卷积核大小
                                                  num_filters=conv_num1, # 卷积核数量
                                                  conv_stride=1, # 卷积步长=1
                                                  conv_padding=1, # 卷积填充=1
                                                  pool_size=2, # 2*2区域池化
                                                  pool_stride = 2, # 池化步长
                                                  pool_type='max', # 池化类型最大池化
                                                  act="relu")#激活函数
    s1= conv_pool_1.shape
    # drop = conv_pool_1
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=drop1)
    d1 = drop.shape

    # 第二组
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop, # 以上一个drop输出作为输入
                                                  filter_size=3,  # 卷积核大小
                                                  num_filters=conv_num2,  # 卷积核数量
                                                  conv_stride=1,  # 卷积步长=1
                                                  conv_padding=1,  # 卷积填充=0
                                                  pool_size=2,  # 2*2区域池化
                                                  pool_stride=2,  # 池化步长值
                                                  pool_type='max',  # 池化类型最大池化
                                                  act="relu")  # 激活函数
    s2 = conv_pool_2.shape
    # drop=conv_pool_2
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=drop2)
    d2 = drop.shape

    # 全连接层
    fc = fluid.layers.fc(input=drop, size=fc, act="relu")
    f=fc.shape
    # dropout
    drop = fluid.layers.dropout(x=fc, dropout_prob=drop3)
    d3 = drop.shape
    # 输出层(fc)
    predict = fluid.layers.fc(input=drop, # 输入
                              size=type_size, # 输出值的个数
                              act="softmax") # 输出层采用softmax作为激活函数
    return predict,s1,d1,s2,d2,f,d3

def CNN(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,
        conv_num1,drop1,conv_num2,drop2,fc,drop3,learning_rate):
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
    image = fluid.layers.data(name="image", shape=[channel, row_size, col_size], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # 调用函数，创建CNN
    predict, s1, d1, s2, d2, f, d3 = convolution_neural_network(image, type_size,conv_num1,drop1,conv_num2,drop2,fc,drop3)
    print('s1:', s1, '\td1:', d1, '\ts2:', s2, '\td2:', d2, '\tf:', f, '\td3:', d3)
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
        train_cost = 0
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
        parameter(BATCH_SIZE, BUF_SIZE,conv_num1,drop1,conv_num2,drop2,fc,drop3,learning_rate, parameter_txt)
        print("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f"
              % (pass_id, train_cost, train_acc, test_cost, test_acc))
        result_txt = model_save_dir + '/train_process.txt'
        with open(result_txt, "a") as f:
            f.write("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f\n" % (
                pass_id, train_cost, train_acc, test_cost, test_acc))
        # 训练过程可视化
        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)
        # 训练结束后，保存模型
        model_save_path = model_save_dir + '/CNN2D_' + str(pass_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        fluid.io.save_inference_model(dirname=model_save_path,
                                      feeded_var_names=["image"],
                                      target_vars=[predict],
                                      executor=exe)
    end = time()
    time = end - start
    print("用时%fs" % time)

    print('CNN2D:\t' + 'batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE) + '\tend!')



