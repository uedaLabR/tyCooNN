import pandas as pd
from tensorflow import keras
import nnmodels.CNNWavenetBaseCall as cnnwavenetB
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
def evaluate(paramPath,indirs,outdir,outpath,fasta,fasta5out):

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

    model = cnnwavenetB.build_network(shape=(None, param.trimlen, 1), num_classes=len(trnas))
    model.load_weights(outweight)

    totalcounter = Counter(trnas)
    cnt = 0
    fqpath = outpath + "/trna.fastq"
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    fq = open(fqpath, mode='w')
    for f5file in f5list:
        counter = evaluateEach(param,f5file,outpath,model,trnas,fasta,fasta5out,cnt,fq)
        totalcounter.sumup(counter)
        cnt +=1
        print("doing..{}/{}".format(cnt,len(f5list)))

    fq.close()

    #output result
    csvout = outpath + "/count.csv"
    data = []
    data.append(totalcounter.passfilterCnt)
    data.append(totalcounter.allCnt)

    df = pd.DataFrame(data, columns=trnas)
    df.to_csv(csvout)
    #
    filtercsv = outpath + "/filer.csv"
    data = []
    data.append(totalcounter.filterFlgCnt)
    filterlabel = ["pass","meanqlow","siglen","adap2fail","deltalow","delthigh","readlenlow","readlenhigh","trimfail"]
    df = pd.DataFrame(data, columns=filterlabel)
    df.to_csv(filtercsv)

from Bio import SeqIO
def fastaToDict(fasta):

    seqdict = {}
    for record in SeqIO.parse(fasta, 'fasta'):
        seqdict[record.id]  = record.seq.replace('U','T')

    return seqdict

# do it file by file
def evaluateEach(param,f5file,outpath,model,trnas,fasta,fasta5out,cnt_file,fq):

    reads = ut.get_fast5_reads_from_file(f5file)
    trimmed_filterFlgged_read = tn.trimAdaptor(reads, param)
    format_reads = tn.formatSignal(trimmed_filterFlgged_read, param)
    datalabel = []
    data = []
    datadict = {}

    seqdict = fastaToDict(fasta)

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

    print(len(datalabel))

    data = np.reshape(data, (-1, param.trimlen, 1))
    prediction = model.predict(data, batch_size=None, verbose=0, steps=None)

    cnt = -1
    for row in prediction:

        # incriment
        cnt += 1
        rdata = np.array(row)
        maxidxs = np.where(rdata == rdata.max())
        #unique hit with more than zero Intensity
        if len(maxidxs) == 1 and rdata.max() > 0.5:
            maxidx = int(maxidxs[0])
            maxv = rdata.max()
            maxtrna = trnas[maxidx]
            readid = datalabel[cnt]
            minicnt =  datadict[readid]
            minicnt.addInference(maxtrna,maxidx,maxv)

    #
    counter = Counter(trnas)
    for key in datadict:
        minicnt = datadict[key]
        counter.inc(minicnt)

    singlefast5dir = outpath + "/single_fast5"

    #output fast5
    if fasta5out != "None":
        single5out = "S" in fasta5out
        copyWithAdddata(f5file,fast5out,datadict,seqdict,single5out,singlefast5dir,cnt_file,fq)

    return counter

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