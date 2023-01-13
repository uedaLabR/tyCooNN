from pyarrow import parquet as pq
import numpy as np
from multiprocessing import Pool
import random
import glob
import pandas as pd
from tensorflow import keras
import modbasecall.CNNWavenetBaseCallPart as CNNPart
from numba import jit
import multiprocessing
import csv
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import utils.tyUtils as ut
from training.SignalGenerator import ArgumentlGeneratorMatrix
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow

# @click.option('--inp')
# @click.option('--out')
# @click.option('--extn', default='.pq')
# @click.option('--test', type=float, default=0.1)
# @click.option('--seed', type=int, default=100)
# @click.option('--limit', type=int, default=10000)
# @click.option('--ngpu', type=int, default=2)
# @click.option('--epoch', type=int, default=500)
# @click.option('--batch', type=int, default=64)
# def train(inp, out, extn, test, seed, limit, ngpu, epoch, batch):
#     CallTrainer(inp, out, extn=extn, testFraction=test, seed=seed, limit=limit,
#                 wlen=4096, pseudocount=0.1, gpu_count=ngpu, epoch=epoch, batch_size=batch)


def prepare_data(df_inp,trnas):

    X = []
    Xtrace = []
    Y = []

    # pqt = pq.read_table(f,
    #   columns=['read_id', 'trna','meanq','normdelta','countCp','trimsignal', 'trimtrace','fastq'])

    for idx, row in df_inp.iterrows():

        trna = row[1]
        signal = row[5]
        trace = row[6]

        index = trnas.index(trna)

        X.append(signal)
        Xtrace.append(trace)
        Y.append(index)

    X = np.array(X)
    Y = np.array(Y)
    return X,Xtrace,Y

def formatX(X,wlen):
   return np.reshape(X, (-1, wlen, 1))

def formatY(Y):
   Y = np.reshape(Y, (-1, 99,34))
   return Y

def toNumdic(trnas,labeldic):

    rd = {}
    n = 0
    for trna in trnas:

        rd[n] = labeldic[trna]
        n += 1
    return rd

def poswitoutNull(pos, na):

    num = 0
    for n in range(pos):

        a = na[n]
        if np.sum(a) == 0:
            continue
        num +=1

    return num

def formatX(X,wlen):
   return np.reshape(X, (-1, wlen, 1))

def formatY(Y,seqlen):
   Y = np.reshape(Y, (-1, seqlen,34))
   return Y

import os
from modbasecall.ExtrectReader import RowReader
def train(dirpaths,outdir,labeldic,portion,epoch = 50,data_argument = 0):

    wlen = 1024
    space = 19
    if portion < 9:
        start = portion*5
        end = start+10
    elif portion == 8 or portion == 9:
        start = portion*5
        end = start + space +10
    elif portion > 9 and portion < 15:
        start = portion*5 +space
        end = start + 10
    elif portion == 15:
        start = 48
        end = start + space


    fl = []
    trnas = []
    fs = glob.glob(dirpaths[0] + "/*.pq*")
    for f in fs:
        fl.append(f)
        bname = os.path.basename(f)
        bname = bname.replace(".pq","").lower()
        trnas.append(bname)

    fs = glob.glob(dirpaths[1] + "/*.pq*")
    for f in fs:
        fl.append(f)
        bname = os.path.basename(f)
        bname = bname.replace(".pq","").lower()
        trnas.append(bname)

    #get KO parquet
    for name in labeldic:
        ns = name.split("_")
        if len(ns)!=2:
            continue
        if ns[1] =="ivt":
            continue
        trnas.append(name)
        tRNA = ns[0].capitalize()
        if tRNA == "Tyr":
            tRNA = "Tyr1"
        if tRNA == "Ala1":
            tRNA = "Ala1B"
        if tRNA == "Fmet":
            tRNA = "fMet1"
        if tRNA == "Val2a":
            tRNA = "Val2A"
        if tRNA == "Val2b":
            tRNA = "Val2B"

        path = dirpaths[2]+"_"+ns[1]+"/"+tRNA+".pq"
        fl.append(path)

    # print(fl)
    # print(trnas)
    X = []
    Y = []
    n = 0
    print(trnas)
    for f in fl:
        print(n, f)
        trna = trnas[n]

        n+=1
        #label
        na = labeldic[trna]
        label = na[start:end,:]
        reader =  RowReader(f)
        start = poswitoutNull(start,na)
        end = poswitoutNull(end, na)
        data = reader.getRowData(start,end, takecnt=2400)
        xlen = len(data)
        print(xlen)
        #
        X.extend(data)
        for l in range(xlen):
            Y.append(label)


    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    m = 0
    for x in X:

        y = Y[m]
        if m % 12 >= 2:
            X_train.append(x)
            Y_train.append(y)
        else:
            X_test.append(x)
            Y_test.append(y)
        m+=1

    seqsize = 10
    lr = 0.008
    model = CNNPart.build_network(shape=(None, wlen, 1) , seqsize=seqsize)
    optim = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    batch_size = 256

    outweight = outdir+"/"+str(portion)+".hdf"
    modelCheckpoint = ModelCheckpoint(filepath=outweight,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='max',
                                      period=1)

    historypath = outdir +"/"+str(portion)+'history.csv'
    model.compile(loss='binary_crossentropy',
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                  # optimizer=opt,
                  metrics=['accuracy'])

    test_x = formatX(X_test, wlen)
    test_y = formatY(Y_test,seqsize)
    train_x = formatX(X_train,wlen)
    train_y = formatY(Y_train,seqsize)
    history = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, verbose=1,
              shuffle=True, validation_data=(test_x, test_y),callbacks=[modelCheckpoint])

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(historypath)

