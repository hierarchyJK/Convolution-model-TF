# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2018-12-13 20:04:26
@month:十二月
"""
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
import cnn_utils
np.random.seed(1)
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()
####Example of a picture
#index = 6
#plt.imshow(X_train_orig[index])
#plt.show()
#print('y = ',str(np.squeeze(Y_train_orig[:,index])))

X_train = X_train_orig/255#####对训练数据标准化
X_test = X_test_orig/255
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig,C = 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig,C = 6).T
print("number of training example = "+ str(X_train.shape[0]))##1080
print("number of testing example = "+ str(X_test.shape[0]))##120
print("X_train shape:" + str(X_train.shape))##(1080,64,64,3)
print("Y_train shape:" + str(Y_train.shape))##(1080,6)
print("X_test shape:" + str(X_test.shape))##(120,64,64,3)
print("Y_test shape:" + str(Y_test.shape))##(120,6)


def create_placeholders(n_H0,n_W0,n_C0,n_y):
    """
    为session创建占位符
    :param n_H0: 输入图像的高度
    :param n_W0: 输入图像的宽度
    :param n_C0: 输入的通道数
    :param n_y: 分类数目
    :return:
    X:placeholder for the data input
    Y:placeholder for the input label
    """
    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])
    return X,Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1',[4,4,3,8],initializer=tf.random_normal_initializer(seed= 0))
    W2 = tf.get_variable('W2',[2,2,8,16],initializer=tf.random_normal_initializer(seed = 0))
    parameters = {'W1':W1,'W2':W2}

    return parameters

def forward_propagation(X,parameters):
    """
    Implemnt the forward propagation for the model:
    CONV2D->RELU->MAXPOOL->  CONV2D->MAXPOOL->  FLATTEN->FULLCONNECTION
    :param X:input dataset placeholder;(input size,number of examples)
    :param parameters:W1 and W2
    :return:
    Z3: the output of the last linear unit
    """
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    #RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    # FLATTEN
    P = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn = None)
    return Z3

def computer_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    return cost

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.009,num_epochs = 100,minibatch_size = 64
          ,print_cost = True):
    ops.reset_default_graph()#能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)#确保数据一样
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    X,Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = computer_cost(Z3,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)#获取数据库块的数量
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _,temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            if print_cost == True and epoch % 5 == 0:
                print('当前是第' + str(epoch) + '代，成本值为：' + str(minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)


        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration(per tens)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()

        #####计算Accuracy
        predict_op = tf.arg_max(Z3,1)######1代表行，每行中最大数的位子，也就是one-hot中1的位置
        corrent_prediction = tf.equal(predict_op,tf.arg_max(Y,1))

        accuracy = tf.reduce_mean(tf.cast(corrent_prediction,'float'))
        print(accuracy)
        train_accuracy = accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})

        print('训练集的准确度：'+str(train_accuracy))
        print('测试集的准确度：'+str(test_accuracy))

        return (train_accuracy,test_accuracy,parameters)
_,_,parameters = model(X_train,Y_train,X_test,Y_test)