import preprocess.TrimAndNormalize as tn
from multiprocessing import Pool
import utils.tyUtils as ut
import pandas as pd
import csv


def generatePqForTrainingAll(paramPath,listOfIOPath,takeCount=12000):

    f = open(listOfIOPath, 'r')
    tsv = csv.reader(f, delimiter='\t')

    # tRNAlabel indir   outpq
    for row in tsv:

        tRNALabel, indirs, outpq = row[0],row[1],row[2]
        outpq = outpq.strip(" ")
        outpq = outpq.strip("\t")
        print("doing..",tRNALabel,outpq)
        genaratePqForTraining(paramPath, tRNALabel, indirs, outpq, takeCount)


def genaratePqForTraining(paramPath,tRNALabel,indirs,outpq,takeCount=1200):

    param = ut.get_parameter(paramPath)  # setting of file path and max core

    indirs = indirs.split(",")
    reads = ut.get_fast5_reads_dirs(indirs, param.ncore)
    trimmed_filterFlgged_read = tn.trimAdaptor(reads,param)
    #
    filtered_reads = [read for read in trimmed_filterFlgged_read \
                      if read.filterFlg == 0]
    filtered_reads = filtered_reads[0:takeCount]
    format_reads = tn.formatSignal(filtered_reads,param)
    #
    datalist = []
    for read in format_reads:
        tp = (read.read_id,tRNALabel,read.formatSignal,read.trace,read.move)
        datalist.append(tp)

    df = pd.DataFrame(datalist, columns=['read_id', 'trna', 'trimsignal'])
    df.to_parquet(outpq)

    return reads


