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

def evaluate(paramPath,indirs,outdir,outpath,fasta):

    param = ut.get_parameter(paramPath)
    indirs = indirs.split(",")
    f5list = []
    for dir in indirs:
        f5list.extend(ut.get_fast5_files_in_dir(dir))

    trnapath = outdir + '/tRNAindex.csv'
    trnas = getTRNAlist(trnapath)
    print("trna",trnas)

    model = cnnwavenet.build_network(shape=(None, param.trimlen, 1), num_classes=len(trnas))
    outweight = outdir + "learent_arg_weight.h5"
    if not os.path.exsist():
        outweight = outdir + "/learent_arg_weight.h5"
    model.load_weights(outweight)

    totalcounter = Counter(trnas)
    cnt = 0
    for f5file in f5list:
        counter = evaluateEach(param,f5file,outpath,model,trnas,fasta)
        totalcounter.sumup(counter)
        cnt +=1
        print("doing..{}/{}".format(cnt,len(f5list)))


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
def evaluateEach(param,f5file,outpath,model,trnas,fasta):

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
        if read.trimSuccess:
            datalabel.append(read.read_id)
            data.append(read.formatSignal)

    data = np.reshape(data, (-1, param.trimlen, 1))
    prediction = model.predict(data, batch_size=None, verbose=0, steps=None)

    cnt = -1
    for row in prediction:

        # incriment
        cnt += 1
        rdata = np.array(row)
        maxidxs = np.where(rdata == rdata.max())
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

    #output fast5
    copyWithAdddata(f5file,fast5out,datadict,seqdict)

    return counter

def getDummyQual(seqlen):

    return ''.join(['A' for i in range(seqlen)])

def getFastq(read_id,seqdict,tRNA,seqlen):

    seq = seqdict[tRNA]
    if len(seq) <= seqlen:
        seqlen = len(seq)

    hang = 5
    start = (len(seq)-seqlen)-hang
    if start < 0:
        start = 0
    seq = seq[start:len(seq)]
    qual = getDummyQual(seqlen)
    return "@" + read_id+ " \n"  + seq +"\n" +"+" + "\n" + qual

import logging
import os
import shutil
from ont_fast5_api.fast5_file import Fast5File, Fast5FileTypeError
from ont_fast5_api.multi_fast5 import MultiFast5File
from ont_fast5_api.compression_settings import GZIP
def copyWithAdddata(f5file,fast5out,datadict,seqdict):

    try:
        #copy first
        shutil.copyfile(f5file, fast5out)
        with MultiFast5File(fast5out, 'a') as multi_f5:
            for read in multi_f5.get_reads():

                component = "basecall_1d"
                group_name = "Basecall_1D_099"
                dataset_name ="BaseCalled_template"

                basecall_run = read.get_latest_analysis("Basecall_1D")
                fastq = read.get_analysis_dataset(basecall_run, "BaseCalled_template/Fastq")
                seqlen = len(fastq.split("\n")[1])

                minicnt = datadict[read.read_id]
                fastqadd = getFastq(read.read_id,seqdict,minicnt.tRNA,seqlen)


                attrs = {
                    "tRNA": minicnt.tRNA,
                    "tRNAIndex": minicnt.tRNAIdx,
                    "value": minicnt.maxval,
                    "filterpass": (minicnt.filterFlg==0),
                    "filterflg": minicnt.filterFlg,
                    "trimSuccess": minicnt.trimSuccess
                }
                read.add_analysis(component, group_name, attrs)
                path = 'Analyses/{}/'.format(group_name)
                read.handle[path].create_group(dataset_name)
                path = 'Analyses/{}/{}'.format(group_name,dataset_name)
                read.handle[path].create_dataset('Fastq', data=str(fastqadd))



    except Fast5FileTypeError:
        print("error")
        pass
    except Exception as e:
        print("error2",e)
        pass
