# -*- coding: utf-8 -*-
# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

'''
环境：paddlepaddle必须为1.8gpu版本
卷积神经网络CNN 训练模型，输出损失函数图及准确性图

输入：数据样本所在目录（Addfactors1_7.py生成的结果）/模型保存路径/训练影像行、列大小/通道数/分类结果个数
输出：CNN训练精度和损失值+CNN模型

注意：CNN内的结构和批次大小需要自己在里面修改，得到最优结果，适时停止迭代
     模型每一次迭代都会保存一个以数字（1,2,3......）命名的文件，需要自己选择

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
    img, label = sample
    if not os.path.exists(img):
        print(img, "图片不存在")

    proj, geotrans, img = GRID.read_img(img)
    return img, label

def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.replace("\n","").split("\t")
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(train_mapper,
                                      reader,
                                      cpu_count(),
                                      buffered_size)

def test_mapper(sample):
    img, label = sample
    if not os.path.exists(img):
        print(img, "图片不存在")
    proj, geotrans, img = GRID.read_img(img)
    return img, label

def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.replace("\n","").split("\t")
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), buffered_size)


def convolution_neural_network(image, type_size,conv_num1,drop1,conv_num2,drop2,fc,drop3):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,
                                                  filter_size=3,
                                                  num_filters=conv_num1,
                                                  conv_stride=1,
                                                  conv_padding=1,
                                                  pool_size=2,
                                                  pool_stride = 2,
                                                  pool_type='max',
                                                  act="relu")
    s1= conv_pool_1.shape
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=drop1)
    d1 = drop.shape

    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,
                                                  filter_size=3,
                                                  num_filters=conv_num2,
                                                  conv_stride=1,
                                                  conv_padding=1,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  pool_type='max',
                                                  act="relu")
    s2 = conv_pool_2.shape
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=drop2)
    d2 = drop.shape

    fc = fluid.layers.fc(input=drop, size=fc, act="relu")
    f=fc.shape
    drop = fluid.layers.dropout(x=fc, dropout_prob=drop3)
    d3 = drop.shape
    predict = fluid.layers.fc(input=drop,
                              size=type_size,
                              act="softmax")
    return predict,s1,d1,s2,d2,f,d3

def CNN(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,
        conv_num1,drop1,conv_num2,drop2,fc,drop3,learning_rate):
    from time import time
    paddle.enable_static()
    test_file_path = data_root_path + "test.txt"
    train_file_path = data_root_path + "train.txt"
    trainer_reader = train_r(train_list=train_file_path)
    random_train_reader = paddle.reader.shuffle(reader=trainer_reader,
                                                buf_size=BUF_SIZE)
    batch_train_reader = paddle.batch(random_train_reader,
                                      batch_size=BATCH_SIZE)

    test_reader = test_r(test_list=test_file_path)
    random_test_reader = paddle.reader.shuffle(reader=test_reader,
                                               buf_size=BUF_SIZE)
    batch_test_reader = paddle.batch(random_test_reader,
                                     batch_size=BATCH_SIZE)

    image = fluid.layers.data(name="image", shape=[channel, row_size, col_size], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    predict, s1, d1, s2, d2, f, d3 = convolution_neural_network(image, type_size,conv_num1,drop1,conv_num2,drop2,fc,drop3)
    print('s1:', s1, '\td1:', d1, '\ts2:', s2, '\td2:', d2, '\tf:', f, '\td3:', d3)

    cost = fluid.layers.cross_entropy(input=predict,
                                      label=label)
    avg_cost = fluid.layers.mean(cost)

    accuracy = fluid.layers.accuracy(input=predict,
                                     label=label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    optimizer.minimize(avg_cost)

    # place = fluid.CPUPlace() #CPU训练
    place = fluid.CUDAPlace(0)  # GPU训练
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    feeder = fluid.DataFeeder(feed_list=[image, label],
                              place=place)

    test_costs, test_accs = [], []
    train_costs, train_accs = [], []
    testing_costs, testing_accs = [], []
    training_costs, training_accs = [], []
    train_times = 0
    test_times = 0
    train_batches = []
    test_batches = []

    print('开始训练...')
    start = time()
    for pass_id in range(15000):
        train_cost = 0
        train_times += 1
        for batch_id, data in enumerate(batch_train_reader()):
            train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, accuracy])
            training_accs.append(train_acc[0])
            training_costs.append(train_cost[0])
        train_batches.append(train_times)
        train_cost = sum(training_costs) / len(training_costs)
        train_acc = sum(training_accs) / len(training_accs)
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
        test_cost = sum(testing_costs) / len(testing_costs)
        test_acc = sum(testing_accs) / len(testing_accs)
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

        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)

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



