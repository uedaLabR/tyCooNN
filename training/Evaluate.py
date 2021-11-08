from pyarrow import parquet as pq
import numpy as np
from multiprocessing import Pool
import random
import glob
import pandas as pd
from tensorflow import keras
import nnmodels.CNNWavenet as cnnwavenet
from numba import jit
import multiprocessing
import csv
import numpy as np
import os


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



def evaluate(dirpath,outdir,csvout):

    fs = glob.glob(dirpath + "/*.pq*")
    trnas = []

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for f in fs:

        pqt = pq.read_table(f,
                            columns=['read_id', 'trna','trimsignal'])

        dfp = pqt.to_pandas()
        cnt = 0
        wlen = 0
        for idx, row in dfp.iterrows():
            trna = row[1]
            signal = row[2]
            if wlen ==0:
                wlen = len(signal)

            if cnt % 12 >= 2:
                X_train.append(signal)
                Y_train.append(trna)
            else:
                X_test.append(signal)
                Y_test.append(trna)

            cnt+=1

        trna = dfp["trna"].unique()
        trnas.append(trna)

    trnas = sorted(trnas)
    # name to index
    Y_train = list(map(lambda trna: trnas.index(trna), Y_train))
    Y_test = list(map(lambda trna: trnas.index(trna), Y_test))


    num_classes = np.unique(Y_train).size
    #

    test_x = np.reshape(X_test, (-1, wlen, 1))
    testy_noncategorical = Y_test


    #model = wavenet.build_network(shape=(None, wlen, 1), num_classes=num_classes)
    model = cnnwavenet.build_network(shape=(None, wlen, 1), num_classes=num_classes)

    outweight = outdir + "learent_arg_weight.h5"
    if not os.path.exsist():
        outweight = outdir + "/learent_arg_weight.h5"

    model.load_weights(outweight)
    #

    prediction = model.predict(test_x, batch_size=None, verbose=0, steps=None)
    probthres = 0.75
    retdict = {}
    cnt = -1
    for row in prediction:

        # incriment
        cnt += 1
        rdata = np.array(row)
        if rdata.max() < probthres:
            continue
        maxidxs = np.where(rdata == rdata.max())
        ans = testy_noncategorical[cnt]
        if len(maxidxs) > 1:
            continue  # multiple hit
        maxidx = maxidxs[0]

        if ans in retdict:
            ridxs = retdict[ans]
            ridxs[maxidx] = ridxs[maxidx] + 1
        else:
            ridxs = np.zeros(num_classes)
            retdict[ans] = ridxs
            ridxs[maxidx] = ridxs[maxidx] + 1

    #
    data = []
    title = []
    title.append("/")
    title.extend(trnas)
    for i in range(num_classes):
        trna = trnas[i]
        line = []
        line.append(trna)
        line.extend(list(retdict[i]))
        data.append(line)

    df = pd.DataFrame(data, columns=title)
    df.to_csv(csvout)




