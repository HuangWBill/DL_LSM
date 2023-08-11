# -*- coding: utf-8 -*-

'''
环境：paddlepaddle必须为1.8gpu版本
深度神经网络DNN（多层感知器MLP） 训练模型，输出损失函数图及准确性图

输入：数据样本所在目录（Addfactors1_7.py生成的结果）/模型保存路径/训练影像行、列大小/通道数/分类结果个数
输出：MLP训练精度和损失值+MLP模型

注意：MLP内的结构和批次大小需要自己在里面修改，得到最优结果，适时停止迭代
     模型每一次迭代都会保存一个以数字（1,2,3......）命名的文件，需要自己选择

位置："Addfactors_2.py"+"Layer_stacking_4.py"+"data_pre-processing_3.py"运行完成后进行；后接"ROC-AUC-single_8.py"
'''

import os
from multiprocessing import cpu_count
import paddle
import paddle.fluid as fluid
from Common_use import GRID
from Common_use import others as ots


###########################模型训练评估-MLP################################

def parameter(Batch,Buf,h1_size,drop1,h2_size,drop2,h3_size,drop3,learning_rate,parameter_txt):
    with open(parameter_txt, "w") as f:
        pass
    with open(parameter_txt, "a") as f:
        f.write('BATCH_SIZE = %d\nBUF_SIZE = %d\n\n hidden1=%d,\n drop1=%f\n\n hidden2=%d,\n drop2=%f\n\n'
                'hidden3=%d,\n drop3=%f\n\n learning_rate=%f\n' % (Batch,Buf,h1_size,drop1,h2_size,drop2,h3_size,drop3,learning_rate))

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

# 定义多层感知器
# type_size输出值个数
def multilayer_perceptron(input,type_size,h1_size,drop1,h2_size,drop2,h3_size,drop3):
    hidden1 = fluid.layers.fc(input=input, size=h1_size, act="relu")
    h1 = hidden1.shape
    drop = fluid.layers.dropout(x=hidden1, dropout_prob=drop1)
    d1 = drop.shape
    hidden2 = fluid.layers.fc(input=drop, size=h2_size, act="relu")
    h2 = hidden2.shape
    drop = fluid.layers.dropout(x=hidden2, dropout_prob=drop2)
    d2 = drop.shape
    hidden3 = fluid.layers.fc(input=drop, size=h3_size, act="relu")
    h3 = hidden3.shape
    drop = fluid.layers.dropout(x=hidden3, dropout_prob=drop3)
    d3 = drop.shape
    prediction = fluid.layers.fc(input=drop, size=type_size, act="softmax")
    return prediction,h1,d1,h2,d2,h3,d3

def MLP(type_size,data_root_path,model_save_dir,row_size,col_size,channel,BATCH_SIZE,BUF_SIZE,
        h1_size,drop1,h2_size,drop2,h3_size,drop3,learning_rate):
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

    # 调用MLP
    predict, h1, d1, h2, d2, h3, d3 = multilayer_perceptron(image, type_size, h1_size, drop1, h2_size, drop2, h3_size,drop3)
    print('h1:', h1, '\td1:', d1, '\th2:', h2, '\td2:', d2, '\th3:', h3, '\td3:', d3)
    # 定义损失函数和准确率
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    # 从主程序中克隆一个测试程序
    test_program = fluid.default_main_program().clone(for_test=True)
    # 定义优化函数
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate)
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
        parameter(BATCH_SIZE, BUF_SIZE,h1_size,drop1,h2_size,drop2,h3_size,drop3,learning_rate,parameter_txt)
        print("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f"
              % (pass_id, train_cost, train_acc, test_cost, test_acc))
        result_txt = model_save_dir + '/train_process.txt'
        with open(result_txt, "a") as f:
            f.write("Pass:%d \tTrain_cost:%.5f\tTrain_acc:%.5f\tTest_cost:%.5f\tTest_acc:%.5f\n" % (
                pass_id, train_cost, train_acc, test_cost, test_acc))
        # 训练过程可视化
        ots.dy_fig(train_batches, train_costs, train_accs, test_batches, test_costs, test_accs)
        # 训练结束后，保存模型
        model_save_path = model_save_dir + '/MLP_' + str(pass_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        fluid.io.save_inference_model(dirname=model_save_path,
                                      feeded_var_names=["image"],
                                      target_vars=[predict],
                                      executor=exe)
    end = time()
    time = end - start
    print("用时%fs" % time)

    print('MLP:\t' + 'batch_' + str(BATCH_SIZE) + 'buf_' + str(BUF_SIZE) + '\tend!')


