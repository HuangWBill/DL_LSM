# -*- coding: utf-8 -*-
# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
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
    img, label = sample
    if not os.path.exists(img):
        print(img, "图片不存在")

    proj, geotrans, img = GRID.read_img(img)
    k = 0
    hang = np.zeros((img.shape[0], img.shape[1] * img.shape[2]))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            hang[:, k] = img[:, i, j]
            k = k + 1
    return hang, label

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
    k = 0
    hang = np.zeros(((img.shape[0], img.shape[1] * img.shape[2])))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            hang[:,k] = img[:, i, j]
            k = k + 1
    return hang, label

def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.replace("\n","").split("\t")
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), buffered_size)

def long_short_term_memory(image,type_size,h_size,h_layer,fc_size,drop_size):
    lstm = paddle.nn.LSTM(input_size=image.shape[2],
                          hidden_size=h_size,
                          num_layers=h_layer,
                          direction='forward',
                          dropout=0.0)
    lstm_result, final_states = lstm(inputs=image)

    fc = fluid.layers.fc(input=lstm_result, size=fc_size, act="relu")

    drop = fluid.layers.dropout(x=fc, dropout_prob=drop_size)

    predict = fluid.layers.fc(input=drop,
                              size=type_size,
                              act="softmax")
    return predict

def LSTM(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,
         h_size,h_layer,fc_size,drop_size,learning_rate):
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

    image = fluid.layers.data(name="image", shape=[channel, row_size * col_size], dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    predict = long_short_term_memory(image, type_size, h_size, h_layer, fc_size, drop_size)

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
        parameter(BATCH_SIZE, BUF_SIZE,h_size,h_layer,fc_size,drop_size,learning_rate, parameter_txt)
        print("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f"
              % (pass_id, train_cost, train_acc, test_cost, test_acc))
        result_txt = model_save_dir + '/train_process.txt'
        with open(result_txt, "a") as f:
            f.write("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f\n" % (
                pass_id, train_cost, train_acc, test_cost, test_acc))

        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)

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




