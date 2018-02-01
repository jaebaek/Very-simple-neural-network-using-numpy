#!/usr/bin/env python
# -*- coding: utf-8 -*-

import parser
import sys
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cross_entropy(y, y_hat):
    if y == 0 and y_hat == 1:
        y_hat = 0.9999
    if y == 1 and y_hat == 0:
        y_hat = 0.0001
    return -y*np.log(y_hat) - (1-y)*np.log((1-y_hat))

labels = parser.do_parse('train-labels-idx1-ubyte', parser.TrainLabel())
images = parser.do_parse('train-images-idx3-ubyte', parser.TrainImage())

print labels[0]
print images[0]

n_pix = len(images[0])
n_hid = 20
n_y = 10

v = np.random.rand(n_pix, n_hid)
w = np.random.rand(n_y, n_hid)
bj = np.random.rand(n_hid)
b = np.random.rand(n_y)
learn_rate = 0.01
i = 0
for k in range(100):
    err = 0
    n = 0
    for x, l in zip(images, labels):
        y = 1 if l == i else 0
        zj = np.dot(x, v) + bj
        aj = sigmoid(zj)

        z = np.dot(aj, w[i]) + b[i]
        a = sigmoid(z)

        err = err + cross_entropy(y, a)

        grad_z = -(y-a)
        grad_w = aj * grad_z
        grad_b = grad_z
        grad_v = grad_z * np.dot(np.array([x]).T, np.array([w[i]]))
        grad_bj = w[i] * grad_z

        w = w - learn_rate * grad_w
        v = v - learn_rate * grad_v
        b = b - learn_rate * grad_b
        bj = bj - learn_rate * grad_bj

        n = n + 1
        if n == 1000:
            break
    print err / n

print '-------------------------------'

def check_correct(y, y_hat):
    if y == 1:
        if y_hat > 0.6:
            return True
        else:
            return False
    if y == 0:
        if y_hat < 0.4:
            return True
        else:
            return False

n = 0
good = 0
bad = 0
for x, l in zip(images, labels):
    y = 1 if l == i else 0
    zj = np.dot(x, v) + bj
    aj = sigmoid(zj)

    z = np.dot(aj, w[i]) + b[i]
    a = sigmoid(z)

    if check_correct(y, a):
        good = good + 1
    else:
        bad = bad + 1

    n = n + 1
    if n == 1000:
        print good, bad
        good, bad = 0, 0
print good, bad
