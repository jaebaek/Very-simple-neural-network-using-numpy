#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

def readInt(fd, n):
    rtn = 0
    for i in range(0, n):
        s = fd.read(1)
        if s == '':
            return 0
        rtn = rtn * 0x100 + int(ord(s))
    return rtn

def readArray(fd, nelem, nbyte):
    rtn = []
    for i in range(0, nelem):
        rtn.append(readInt(fd, nbyte))
    return rtn

def readMatrix(fd, nrow, ncol):
    mat = []
    for i in range(0, nrow):
        mat = mat + readArray(fd, ncol, 1)
    return mat

def readMatrixArray(fd, nelem, nrow, ncol):
    rtn = []
    for i in range(0, nelem):
        rtn.append(np.array(readMatrix(fd, nrow, ncol)))
    return rtn

class MNISTFile:
    def __init__(self):
        pass

    def parse(self, fd):
        return ''

class TrainLabel(MNISTFile):
    def __init__(self):
        pass

    def parse(self, fd):
        # check magic number
        magic = readInt(fd, 4)
        assert (magic == 2049)

        # number of items
        n = readInt(fd, 4)
        assert (n == 60000)

        # read labels
        return readArray(fd, n, 1)

class TrainImage(MNISTFile):
    def __init__(self):
        pass

    def parse(self, fd):
        # check magic number
        magic = readInt(fd, 4)
        assert (magic == 2051)

        # number of items
        n = readInt(fd, 4)
        assert (n == 60000)
        n = 2000

        # number of row
        nrow = readInt(fd, 4)
        assert (nrow == 28)

        # number of column
        ncol = readInt(fd, 4)
        assert (ncol == 28)

        # read labels
        return readMatrixArray(fd, n, nrow, ncol)

def do_parse(file_name, mnist):
    f = open(file_name, 'rb')
    rtn = mnist.parse(f)
    f.close()
    return rtn
