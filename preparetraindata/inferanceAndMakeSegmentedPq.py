import pandas as pd
from tensorflow import keras
import nnmodels.CNNWavenet as cnnwavenet
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


def getKey(species):
    margeMap = {"ala1": "Ala1B", "fmet": "fMet1", "tyr": "Tyr1"}
    if species in margeMap:
        return margeMap[species]
    return species

def getTRNAlist(trnapath):

    trnas = []
    with open(trnapath) as f:
        l = f.readlines()
        for trna in l:
            if len(trna) > 0:
                trna = trna.replace('\n','')
                trna = trna.replace('\"', '')
                trna = getKey(trna)
                trnas.append(trna)
    return trnas

import os.path
def evaluate(paramPath,indirs,outdir,outpath,postfix):

    outweight = outdir + "learent_arg_weight.h5"
    if not os.path.isfile(outweight):
        outweight = outdir + "/learent_arg_weight.h5"

    param = ut.get_parameter(paramPath)
    indirs = indirs.split(",")
    f5list = []
    for dir in indirs:
        f5list.extend(ut.get_fast5_files_in_dir(dir))

    trnapath = outdir + '/tRNAindex.csv'
    trnas = getTRNAlist(trnapath)
    print("trna",trnas)

    model = cnnwavenet.build_network(shape=(None, param.trimlen, 1), num_classes=len(trnas))
    model.load_weights(outweight)

    cnt = 0
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    datalist = []
    for f5file in f5list:
        r1 = evaluateEachFile(param,f5file,outpath,model,trnas,postfix)
        datalist.extend(r1)
        cnt +=1
        print("doing..{}/{}".format(cnt,len(f5list)))
        print("lendata", len(datalist))


    return datalist

from Bio import SeqIO
def fastaToDict(fasta):

    seqdict = {}
    for record in SeqIO.parse(fasta, 'fasta'):
        seqdict[record.id]  = record.seq.replace('U','T')

    return seqdict


# viterbi segmentation
import numpy as np
# do it file by file
def evaluateEachFile(param,f5file,outpath,model,trnas,postfix):

    reads = ut.get_fast5_reads_from_file(f5file)
    trimmed_filterFlgged_read = tn.trimAdaptor(reads, param)
    format_reads = tn.formatSignal(trimmed_filterFlgged_read, param)
    reads = []
    data = []
    datadict = {}

    fast5dir = outpath +"/fast5"
    if not os.path.exists(fast5dir):
        os.makedirs(fast5dir)

    for read in format_reads:

        datadict[read.read_id] = MiniCounter(read.filterFlg,read.trimSuccess)
        #print(read.read_id)
        if (read.filterFlg == 0):
            reads.append(read)
            data.append(read.formatSignal)


    fdata = np.reshape(data, (-1, param.trimlen, 1))
    prediction = model.predict(fdata, batch_size=None, verbose=0, steps=None)

    cnt = -1
    readsret = []
    for row in prediction:

        # incriment
        cnt += 1
        rdata = np.array(row)
        maxidxs = np.where(rdata == rdata.max())
        #unique hit with more than zero Intensity
        if len(maxidxs) == 1 and rdata.max() > 0.5:
            maxidx = int(maxidxs[0])
            maxtrna = trnas[maxidx]
            read = reads[cnt]
            read.inferencedtRNA = maxtrna
            readsret.append(read)

            #print("sig",read.normSig)

    return readsret


def getDummyQual(seqlen):

    return ''.join(['A' for i in range(seqlen)])

def getFastq(read_id,seqdict,tRNA,seqlen):

    if tRNA not in seqdict:
        #print(tRNA)
        return None

    seq = seqdict[tRNA]
    if len(seq) <= seqlen:
        seqlen = len(seq)

    hang = 5
    start = (len(seq)-seqlen)-hang
    if start < 0:
        start = 0
    #seq = seq[start:len(seq)]
    qual = getDummyQual(len(seq))
    fq = str(read_id)+ " \n"  + str(seq) +"\n" +"+" + "\n" + str(qual)
    #print(fq)
    return fq

import logging
import os
import shutil
from ont_fast5_api.fast5_file import Fast5File, Fast5FileTypeError
from ont_fast5_api.multi_fast5 import MultiFast5File
from ont_fast5_api.compression_settings import GZIP
import ont_fast5_api.conversion_tools.multi_to_single_fast5 as multi_to_single_fast5
import h5py
import sys
if sys.version_info[0] > 2:
    unicode = str


