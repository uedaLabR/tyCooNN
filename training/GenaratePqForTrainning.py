import preprocess.TrimAndNormalize as tn
from multiprocessing import Pool
import utils.tyUtils as ut
import pandas as pd
import csv


def generatePqForTrainingAll(paramPath,listOfIOPath,takeCount=12000):

    f = open(listOfIOPath, 'r')
    tsv = csv.reader(f, delimiter='\t')

    # tRNAlabel indir   outpq
    stat = {}
    for row in tsv:

        tRNALabel, indirs, outpq = row[0],row[1],row[2]
        outpq = outpq.strip(" ")
        outpq = outpq.strip("\t")
        print("doing .. ",tRNALabel,outpq)
        read,s = genaratePqForTraining(paramPath, tRNALabel, indirs, outpq, takeCount)
        stat[tRNALabel] = s

    return stat

def genaratePqForTraining(paramPath,tRNALabel,indirs,outpq,takeCount=12000):

    param = ut.get_parameter(paramPath)  # setting of file path and max core

    indirs = indirs.split(",")
    reads = ut.get_fast5_reads_dirs(indirs, param.ncore)
    trimmed_filterFlgged_read = tn.trimAdaptor(reads,param)
    
    # Generate throughput output:
    stat = get_filtering_stat(trimmed_filterFlgged_read)

    filtered_reads = [read for read in trimmed_filterFlgged_read \
                      if read.filterFlg == 0]
    filtered_reads = filtered_reads[0:takeCount]
    format_reads = tn.formatSignal(filtered_reads,param)
    #
    datalist = []
    for read in format_reads:
        tp = (read.read_id,tRNALabel,read.formatSignal)
        datalist.append(tp)

    df = pd.DataFrame(datalist, columns=['read_id', 'trna', 'trimsignal'])
    df.to_parquet(outpq)

    return reads,stat

def get_filtering_stat(reads):
    
    stat = {}
    stat['N'] = len(reads)
    stat['meanq'] = sum([1 for read in reads if read.filterFlg == 1])
    stat['maxsignallen'] = sum([1 for read in reads if read.filterFlg == 2])
    stat['maxdurationRate'] = sum([1 for read in reads if read.filterFlg == 3])
    stat['mindelta'] = sum([1 for read in reads if read.filterFlg == 4])
    stat['maxdelta'] = sum([1 for read in reads if read.filterFlg == 5])
    stat['minreadlen'] = sum([1 for read in reads if read.filterFlg == 6])
    stat['maxreadlen'] = sum([1 for read in reads if read.filterFlg == 7])
    stat['trimfail'] = sum([1 for read in reads if read.filterFlg == 8])
    stat['pass'] = sum([1 for read in reads if read.filterFlg == 0])
    print("N: %d, MeanQ: %d, Maxsignallen: %d, MaxdurationRate %d,\n Delta(Min,Max): %d %d, Readlen(Min,Max): %d %d,\n TraimFail %d, Pass: %d\n" \
            % (stat['N'],stat['meanq'],stat['maxsignallen'],stat['maxdurationRate'],stat['mindelta'],stat['maxdelta'],stat['minreadlen'],  \
               stat['maxreadlen'],stat['trimfail'],stat['pass']))
    return stat
