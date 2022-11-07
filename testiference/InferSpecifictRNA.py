import pandas as pd
from tensorflow import keras
import nnmodels.CNNWavenetBaseCall as cnnwavenet
from numba import jit
import multiprocessing
import csv
import numpy as np
import utils.tyUtils as ut
import os
from ont_fast5_api.fast5_interface import get_fast5_file
from inference.ExCounter import Counter
from inference.ExCounter import MiniCounter
import preprocess.TrimAndNormalize as tn


def getTRNAlist(trnapath):

    trnas = []
    with open(trnapath) as f:
        l = f.readlines()
        for trna in l:
            if len(trna) > 0:
                trna = trna.replace('\n','')
                trna = trna.replace('\"', '')
                trnas.append(trna)
    return trnas

import os.path
import utils.labelMatrixUtil as lutil
from utils.GraphManager import GraphManager
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(indirs,outdir,labeldic,modlist,paramPath,weightpath):



    param = ut.get_parameter(paramPath)
    indirs = indirs.split(",")
    f5list = []
    for dir in indirs:
        f5list.extend(ut.get_fast5_files_in_dir(dir))


    model = cnnwavenet.build_network(shape=(None, param.trimlen, 1), num_classes=0)
    model.load_weights(weightpath)

    data = []
    for f5file in f5list:

        reads = ut.get_fast5_reads_from_file(f5file)
        trimmed_filterFlgged_read = tn.trimAdaptor(reads, param)
        format_reads = tn.formatSignal(trimmed_filterFlgged_read, param)
        for read in format_reads:
            signal = read.formatSignal
            wlen = len(signal)
            if wlen == 8192:
                data.append(read.formatSignal)

    data = np.reshape(data, (-1, param.trimlen, 1))
    prediction = model.predict(data, batch_size=None, verbose=0, steps=None)

    cnt = -1
    thr4 = None
    pro3 = None
    for scorematrix in prediction:

        py = lutil.getMaxCandidate(scorematrix, labeldic)
        if py == "thr4":
            if thr4 is None:
                thr4 = scorematrix
            else:
                thr4 = thr4 + scorematrix

        if py == "pro3":

            if pro3 is None:
                pro3 = scorematrix
            else:
                pro3 = pro3 + scorematrix


    gp = outdir + "/pro_thr4_graph"
    gm = GraphManager(gp)
    numl = list(range(1, 100))

    df = pd.DataFrame(pro3, columns=modlist, index=numl)
    fig = plt.figure(figsize=(60, 60))
    sns.heatmap(df, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title("pro3")
    gm.add_figure(fig)

    df = pd.DataFrame(thr4, columns=modlist, index=numl)
    fig = plt.figure(figsize=(60, 60))
    sns.heatmap(df, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title("thr4")
    gm.add_figure(fig)

    gm.save()