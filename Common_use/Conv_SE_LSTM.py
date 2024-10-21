# -*- coding: utf-8 -*-
# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
'''
环境：paddlepaddle必须为2.0gpu版本
'''

import os
from multiprocessing import cpu_count
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from Common_use import others as ots
from Common_use import GRID

def parameter(Batch,Buf,conv_num,channel,h_size,h_layer,fc_size,drop_size,learning_rate,parameter_txt):
    with open(parameter_txt, "w") as f:
        pass
    with open(parameter_txt, "a") as f:
        f.write('BATCH_SIZE = %d\nBUF_SIZE = %d\n\nnum_filters=%d\nfilter_size=3\nstride=2\npadding=0\n'
                'groups=%d\n\nhidden_size=%d\nhidden_layers_num=%d\n\n'
                'fc_size=%d\ndropout=%f\n\nlearning_rate=%f' % (Batch,Buf,conv_num,channel,h_size,h_layer,fc_size,drop_size,learning_rate))

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


class SE_module(nn.Layer):
    def __init__(self, channel, height, width, reduction):
        super(SE_module, self).__init__()
        self.avgpool = nn.AvgPool2D((height, width), 1)
        self.fc_layer = nn.Sequential(
            nn.Linear(channel, int(channel/reduction)),
            nn.ReLU(),
            nn.Linear(int(channel/reduction), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        s_x = self.avgpool(x).reshape([b, c])
        e_x = self.fc_layer(s_x).reshape([b, c, 1, 1])
        refined_feature = x * e_x
        return refined_feature


def Conv_SE_LSTM_frame(image,type_size,conv_num,channel,h_size,h_layer,fc_size,drop_size):

    conv_1 = fluid.layers.conv2d(input=image,
                                 num_filters=conv_num*channel,
                                 filter_size=3,
                                 stride=2,
                                 padding=0,
                                 groups=channel,
                                 act="relu",
                                 data_format="NCHW")
    b, c, h, w = conv_1.shape

    se = SE_module(c, h, w, channel)
    refined_feature = se.forward(conv_1)

    names = locals()
    for i in range(channel):
        names['slcon' + str(i)] = paddle.slice(refined_feature, axes=[1], starts=[conv_num*i], ends=[conv_num*(i+1)])
        names['slcon' + str(i)]=paddle.reshape(names['slcon' + str(i)], (-1, 1, conv_num*refined_feature.shape[2]*refined_feature.shape[3]))
    fconv_1 = names['slcon' + str(0)]
    for i in range(channel-1):
        i=i+1
        fconv_1 = paddle.concat([fconv_1, names['slcon' + str(i)]],axis=1)

    lstm = paddle.nn.LSTM(input_size=fconv_1.shape[2],
                          hidden_size=h_size,
                          num_layers=h_layer,
                          direction='forward',
                          dropout=0.0)
    lstm_result, final_states = lstm(inputs=fconv_1)

    fc = fluid.layers.fc(input=lstm_result, size=fc_size, act="relu")

    drop = fluid.layers.dropout(x=fc, dropout_prob=drop_size)

    predict = fluid.layers.fc(input=drop,
                              size=type_size,
                              act="softmax")
    return predict

def Conv_SE_LSTM(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,learning_rate,conv_num=3,h_size=20,h_layer=2,fc_size=200,drop_size=0.5):
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

    predict = Conv_SE_LSTM_frame(image,type_size,conv_num,channel,h_size,h_layer,fc_size,drop_size)

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
    for pass_id in range(1500):
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
        parameter(BATCH_SIZE, BUF_SIZE,conv_num,channel,h_size,h_layer,fc_size,drop_size,learning_rate, parameter_txt)
        print("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f"
              % (pass_id, train_cost, train_acc, test_cost, test_acc))
        result_txt = model_save_dir + '/train_process.txt'
        with open(result_txt, "a") as f:
            f.write("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f\n" % (
                    pass_id, train_cost, train_acc, test_cost, test_acc))

        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)

        model_save_path = model_save_dir + '/model_' + str(pass_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        fluid.io.save_inference_model(dirname=model_save_path,
                                      feeded_var_names=["image"],
                                      target_vars=[predict],
                                      executor=exe)
    end = time()
    time = end - start
    print("用时%fs" % time)

    print('Conv_Attention_LSTM:\t' + 'batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE) + '\tend!')


