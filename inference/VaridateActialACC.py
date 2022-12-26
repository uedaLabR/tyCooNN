from pyarrow import parquet as pq
import numpy as np
from multiprocessing import Pool
import random
import glob
import pandas as pd
from tensorflow import keras
import nnmodels.CNNWavenetBaseCall as cnnwavenet
from numba import jit
import multiprocessing
import csv
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import utils.tyUtils as ut
from training.SignalGenerator import ArgumentlGenerator
from training.SignalGenerator import BatchIterator
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

import utils.labelMatrixUtil as lutil
import pandas as pd
from utils.GraphManager import GraphManager
import matplotlib.pyplot as plt
import seaborn as sns

def varidate(dirpaths,outdir,labeldic,modlist,samplenames,weightpath):


    fl = []
    for dirpath in dirpaths:
        fs = glob.glob(dirpath + "/*.pq*")
        fl.extend(fs)

    trnas = []

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    wlen = 0
    for f in fl:

        print(f)
        pqt = pq.read_table(f,
                            columns=['read_id', 'trna','trimsignal'])

        dfp = pqt.to_pandas()
        cnt = 0
        wlen = 0
        for idx, row in dfp.iterrows():
            trna = row[1].lower()
            signal = row[2]
            if wlen == 0:
                wlen = len(signal)

            if cnt % 12 >= 2:
                X_train.append(signal)
                Y_train.append(trna)
            else:
                X_test.append(signal)
                Y_test.append(trna)

            if cnt >= 1200:
                break
            cnt+=1

        trna = dfp["trna"].unique()
        trnas.append(trna)

    trnas = sorted(trnas)
    # name to index
    Y_train = list(map(lambda trna: labeldic[trna], Y_train))
    Y_test_str = Y_test
    Y_test = list(map(lambda trna: labeldic[trna], Y_test))

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    Y_train = np.reshape(Y_train, (-1, 99, 34))
    Y_test = np.reshape(Y_test, (-1, 99, 34))

    #use only test data
    lr = 0.0008
    model = cnnwavenet.build_network(shape=(None, wlen, 1), num_classes=0)


    model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                  # optimizer=opt,
                  metrics=['accuracy'])

    model.load_weights(weightpath)

    test_x = formatX(X_test, wlen)

    predictYMatrix = model.predict(test_x)
    predictY = []
    pcnt = 0
    for scorematrix in predictYMatrix:

        py = lutil.getMaxCandidate(scorematrix, labeldic)
        predictY.append(py)

        print(pcnt,len(predictYMatrix))
        pcnt +=1

    l=0
    smd = {}
    for scorematrix in predictYMatrix:

        k = Y_test_str[l]
        if k in smd:
            smd[k] = smd[k] + scorematrix
        else:
            smd[k] = scorematrix

        l = l+1

    gp = outdir + "/graph"
    gm = GraphManager(gp)
    numl = list(range(1,100))
    #tRNAkeys = list(smd.keys())
    tRNAkeys = ["fmet","ile2","leu1","tyr"]
    for k in tRNAkeys:

        v = smd[k]
        df = pd.DataFrame(v,columns=modlist,index=numl)
        fig = plt.figure(figsize=(60, 60))
        sns.heatmap(df, annot=True, fmt='g', cmap='Blues',cbar=False)
        plt.title(k)
        gm.add_figure(fig)

        df = pd.DataFrame(labeldic[k],columns=modlist,index=numl)
        fig = plt.figure(figsize=(60, 60))
        sns.heatmap(df, annot=True, fmt='g', cmap='Blues',cbar=False)
        plt.title(k+"_Ans")
        gm.add_figure(fig)

    gm.save()

    cntdict = {}
    for l,p in zip(Y_test_str,predictY):

        key = str(l) + str(p)

        if key in cntdict:
            cntdict[key] = cntdict[key]+1
        else:
            cntdict[key] = 1

    # keys = sorted(labeldic.keys())
    keys = samplenames
    print(samplenames)

    da = np.zeros((len(keys),len(keys)))
    n =-1
    for key1 in keys:
        n = n+1
        m = -1
        for key2 in keys:
            m = m + 1
            key = str(key1) + str(key2)
            if key in cntdict:
                cnt = cntdict[key]
                da[m][n] = cnt

    df = pd.DataFrame(da,columns=keys,index=keys)
    df.to_csv(outdir+"/result.csv")
