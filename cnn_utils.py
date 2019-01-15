# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2018-12-13 21:01:47
@month:十二月
"""
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
def load_dataset():
    train_dataset = h5py.File("F:\\吴恩达DL作业\\课后作业\\代码作业\\第四课第一周编程作业\\assignment\\datasets\\train_signs.h5","r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set feature
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File("F:\\吴恩达DL作业\\课后作业\\代码作业\\第四课第一周编程作业\\assignment\\datasets\\test_signs.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set feature
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(1,train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

def one_hot_matrix(labels,C):#####通过TensorFlow内在的tf.one_hot()方法实现独热
    C = tf.constant(C,name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

def convert_to_one_hot(Y,C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0):
    m = X.shape[0] #number of training examples==1080
    mini_batches = []
    np.random.seed(seed)

    #step1:Shuffle(X,Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    #step2:mini_batches
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:k * mini_batch_size+mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:k * mini_batch_size+mini_batch_size,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size !=0:
        mini_batch_X = shuffled_X[mini_batch_size * num_complete_minibatches:m,:,:,:]
        mini_batch_Y = shuffled_Y[mini_batch_size * num_complete_minibatches:m,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches