# -*- coding: utf-8 -*-
# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
import matplotlib.pyplot as plt

def dy_fig(train_batches,train_costs,train_accs,test_batches,test_costs,test_accs):
    plt.ion()
    plt.pause(0.1)
    plt.clf()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    plt.sca(ax1)
    plt.plot(train_batches, train_costs)
    plt.title('train_cost')
    plt.sca(ax2)
    plt.plot(train_batches, train_accs, color='r')
    plt.title('train_acc')
    plt.sca(ax3)
    plt.plot(test_batches, test_costs, color='c')
    plt.title('test_cost')
    plt.sca(ax4)
    plt.plot(test_batches, test_accs, color='b')
    plt.title('test_acc')
    plt.ioff()

