# -*- coding: utf-8 -*-

'''
环境：paddlepaddle必须为2.0gpu版本
'''

import os
from multiprocessing import cpu_count
import numpy as np
import paddle
import paddle.fluid as fluid
from Common_use import GRID
from Common_use import others as ots

def parameter(Batch,Buf,h_size,h_layer,fc_size,drop_size,learning_rate,parameter_txt):
    with open(parameter_txt, "w") as f:
        pass
    with open(parameter_txt, "a") as f:
        f.write('BATCH_SIZE = %d\nBUF_SIZE = %d\n\nhidden_size=%d\nhidden_layers_num=%d\n\n'
                'fc_size=%d\ndropout=%f\n\nlearning_rate=%f\n' % (Batch,Buf,h_size,h_layer,fc_size,drop_size,learning_rate))

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
    hang = np.zeros((img.shape[0], img.shape[1] * img.shape[2]))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            hang[:, k] = img[:, i, j]
            k = k + 1
    return hang, label  # 返回图像、类别

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
    # 读取图片内容
    hang = np.zeros(((img.shape[0], img.shape[1] * img.shape[2])))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            hang[:,k] = img[:, i, j]
            k = k + 1
    return hang, label

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

# 搭建LSTM函数
# 结构：输入层 --> 卷积/激活/池化/dropout --> 卷积/激活/池化/dropout -->
#      卷积/激活/池化/dropout --> fc --> dropout --> fc(softmax)
def long_short_term_memory(image,type_size,h_size,h_layer,fc_size,drop_size):
    '''
    循环神经网络计算图
    :param image:输入数据
    :param type_size:输出类别
    :return:
    lstm=paddle.nn.LSTM(input_size (int) - 输入的大小。
                        hidden_size (int) - 隐藏状态大小。
                        num_layers (int，可选) - 网络层数。默认为1。
                        direction (str，可选) - 网络迭代方向，可设置为forward，backward或bidirectional。默认为forward。
                        time_major (bool，可选) - 指定input的第一个维度是否是time steps。默认为False。
                        dropout (float，可选) - dropout概率，指的是出第一层外每层输入时的dropout概率。默认为0。
                        weight_ih_attr (ParamAttr，可选) - weight_ih的参数。默认为None。
                        weight_hh_attr (ParamAttr，可选) - weight_hh的参数。默认为None。
                        bias_ih_attr (ParamAttr，可选) - bias_ih的参数。默认为None。
                        bias_hh_attr (ParamAttr，可选) - bias_hh的参数。默认为None。）
    outputs, final_states(h, c) = lstm(inputs(Tensor):网络输入。如果time_major为True，则Tensor的形状为[time_steps,batch_size,input_size]，如果time_major为False，则Tensor的形状为[batch_size,time_steps,input_size]。
                     (initial_states(tuple,可选):网络的初始状态，一个包含h和c的元组，形状为[num_lauers * num_directions, batch_size, hidden_size]。如果没有给出则会以全零初始化。
                     sequence_length(Tensor,可选) - 指定输入序列的长度，形状为[batch_size]，数据类型为int64或int32。在输入序列中所有time step不小于sequence_length的元素都会被当作填充元素处理（状态不再更新）。))
                输出:outputs(Tensor):输出，由前向和后向cell的输出拼接得到。如果time_major为True，则Tensor的形状为[time_steps,batch_size,num_directions * hidden_size]，如果time_major为False，则Tensor的形状为[batch_size,time_steps,num_directions * hidden_size]，当direction设置为bidirectional时，num_directions等于2，否则等于1。
                    final_states(tuple):最终状态,一个包含h和c的元组。形状为[num_lauers * num_directions, batch_size, hidden_size],当direction设置为bidirectional时，num_directions等于2，否则等于1。
    '''
    print(image.shape)
    # LSTM
    # 这里RNN会有与输入层相同数量的输出层，我们只需要最后一个输出
    lstm = paddle.nn.LSTM(input_size=image.shape[2],
                          hidden_size=h_size,
                          num_layers=h_layer,
                          direction='forward',
                          dropout=0.0)
    lstm_result, final_states = lstm(inputs=image)
    print(lstm_result.shape)
    # 全连接层
    fc = fluid.layers.fc(input=lstm_result, size=fc_size, act="relu")
    print(fc.shape)
    # dropout
    drop = fluid.layers.dropout(x=fc, dropout_prob=drop_size)
    # 输出层(fc)
    predict = fluid.layers.fc(input=drop, # 输入
                              size=type_size, # 输出值的个数
                              act="softmax") # 输出层采用softmax作为激活函数
    return predict

def LSTM(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,
         h_size,h_layer,fc_size,drop_size,learning_rate):
    from time import time
    paddle.enable_static()  # paddlepaddle2.0默认输入动态图，因此如果是静态图需要加这个语句;如果是1.8版本则不需要。
    test_file_path = data_root_path + "test.txt"  # 原测试文件路径
    train_file_path = data_root_path + "train.txt"  # 原训练文件路径
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
    image = fluid.layers.data(name="image", shape=[channel, row_size * col_size], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # 调用函数，创建CNN
    predict = long_short_term_memory(image, type_size, h_size, h_layer, fc_size, drop_size)

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
        parameter(BATCH_SIZE, BUF_SIZE,h_size,h_layer,fc_size,drop_size,learning_rate, parameter_txt)
        print("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f"
              % (pass_id, train_cost, train_acc, test_cost, test_acc))
        result_txt = model_save_dir + '/train_process.txt'
        with open(result_txt, "a") as f:
            f.write("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f\n" % (
                pass_id, train_cost, train_acc, test_cost, test_acc))
        # 训练过程可视化
        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)
        # 训练结束后，保存模型
        model_save_path = model_save_dir + '/IGR_LSTM_' + str(pass_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        fluid.io.save_inference_model(dirname=model_save_path,
                                      feeded_var_names=["image"],
                                      target_vars=[predict],
                                      executor=exe)
    end = time()
    time = end - start
    print("用时%fs" % time)

    print('LSTM:\t' + 'batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE) + '\tend!')




