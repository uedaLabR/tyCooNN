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
        r1 = evaluateEach(param,f5file,outpath,model,trnas,postfix)
        datalist.extend(r1)
        cnt +=1
        print("doing..{}/{}".format(cnt,len(f5list)))
        print("lendata", len(datalist))


    df = pd.DataFrame(datalist, columns=['read_id', 'trna', 'trimsignal'])
    for tRNA in trnas:

        trna_id = tRNA + postfix
        dfselect = df[df.trna == trna_id]
        dfpath = outpath + "/"+trna_id+".pq"
        dfselect.to_parquet(dfpath)

from Bio import SeqIO
def fastaToDict(fasta):

    seqdict = {}
    for record in SeqIO.parse(fasta, 'fasta'):
        seqdict[record.id]  = record.seq.replace('U','T')

    return seqdict

import numpy as np
# do it file by file
def evaluateEach(param,f5file,outpath,model,trnas,postfix):

    reads = ut.get_fast5_reads_from_file(f5file)
    trimmed_filterFlgged_read = tn.trimAdaptor(reads, param)
    format_reads = tn.formatSignal(trimmed_filterFlgged_read, param)
    datalabel = []
    data = []
    datadict = {}

    fast5dir = outpath +"/fast5"
    if not os.path.exists(fast5dir):
        os.makedirs(fast5dir)
    fast5out = fast5dir+"/"+  os.path.basename(f5file)

    for read in format_reads:

        datadict[read.read_id] = MiniCounter(read.filterFlg,read.trimSuccess)
        #print(read.read_id)
        if (read.filterFlg == 0):
            datalabel.append(read.read_id)
            data.append(read.formatSignal)

    print("lendata",len(datalabel))

    fdata = np.reshape(data, (-1, param.trimlen, 1))
    prediction = model.predict(fdata, batch_size=None, verbose=0, steps=None)

    cnt = -1
    ret = []
    for row in prediction:

        # incriment
        cnt += 1
        rdata = np.array(row)
        maxidxs = np.where(rdata == rdata.max())
        #unique hit with more than zero Intensity
        if len(maxidxs) == 1 and rdata.max() > 0.5:
            maxidx = int(maxidxs[0])
            maxv = rdata.max()
            maxtrna = trnas[maxidx]+postfix
            readid = datalabel[cnt]
            minicnt =  datadict[readid]
            minicnt.addInference(maxtrna,maxidx,maxv)
            signal = np.array(data[cnt])
            ret.append((readid,maxtrna,signal))
    #
    return ret

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



import time
def copyWithAdddata(f5file,fast5out,datadict,seqdict,single5out,singlefast5dir,cnt,fq):



    #copy first
    shutil.copyfile(f5file, fast5out)

    with MultiFast5File(fast5out, 'a') as multi_f5:
        rcnt = -1
        for read in multi_f5.get_reads():

            rcnt += 1
            component = "basecall_1d"
            group_name = "Basecall_1D_099"
            dataset_name = "BaseCalled_template"

            basecall_run = read.get_latest_analysis("Basecall_1D")
            fastq = read.get_analysis_dataset(basecall_run, "BaseCalled_template/Fastq")
            # print(fastq)
            seqlen = len(fastq.split("\n")[1])

            #print(read.read_id, (read.read_id in datadict), rcnt)

            if read.read_id in datadict:

                minicnt = datadict[read.read_id]
                fstline = fastq.split("\n")[0]
                fastqadd = getFastq(fstline,seqdict, minicnt.tRNA, seqlen)

                if fastqadd is not None:

                    fq.write(fastqadd)
                    fq.write("\n")

                    attrs = {
                        "tRNA": minicnt.tRNA,
                        "tRNAIndex": minicnt.tRNAIdx,
                        "value": minicnt.maxval,
                        "filterpass": (minicnt.filterFlg == 0),
                        "filterflg": minicnt.filterFlg,
                        "trimSuccess": minicnt.trimSuccess
                    }
                    read.add_analysis(component, group_name, attrs)
                    path = 'Analyses/{}/'.format(group_name)
                    read.handle[path].create_group(dataset_name)
                    path = 'Analyses/{}/{}'.format(group_name, dataset_name)

                    read.handle[path].create_dataset(
                        'Fastq', data=str(fastqadd),
                        dtype=h5py.special_dtype(vlen=unicode))


    multi_f5.close()


    if single5out:
        print('print single5 output to',singlefast5dir,str(cnt+1))
        multi_to_single_fast5.convert_multi_to_single(fast5out, singlefast5dir,str(cnt+1))