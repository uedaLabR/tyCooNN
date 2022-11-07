import glob
import numpy as np
from pyarrow import parquet as pq
import pandas as pd
import utils.tyUtils as ut
import preprocess.TrimAndNormalize as tn
from numpy import mean, absolute
import matplotlib.pyplot as plt

def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)

input = "/share/bhaskar/tRex_211022/result"
keys = ["Ala1B" , "Arg5",  "fMet2",  "Gly2" , "Ile2v" ,   "Leu4",   "Ser1" , "Thr2" , "Tyr2",
"Ala2" ,  "Asn"  ,  "Gln1"  , "Gly3" , "Leu1"  ,  "Leu5" , "Pro1"  , "Ser2" , "Thr3" , "Val1",
"Arg2" ,  "Asp"  ,  "Gln2" ,  "His"  , "Leu1_P" , "Lys"  , "Pro2"  , "Ser3" , "Thr4" , "Val2A",
"Arg3" ,  "Cys"   , "Glu"  ,  "Ile1" , "Leu2"  ,  "Met"  , "Pro3"  , "Ser5" , "Trp"  , "Val2B",
"Arg4" ,  "fMet1" , "Gly1"  , "Ile2" , "Leu3"  ,  "Phe" ,  "Sec" , "Thr1", "Tyr1"]

from utils.GraphManager import GraphManager
gm = GraphManager("/share/trna/tyCooNNTest/test.pdf")
paramPath = '/share/trna/tyCooNN/setting.yaml'
param = ut.get_parameter(paramPath)  # setting of file path and max core
for key in keys:

    datalist = []
    ppathIvt = "/share/bhaskar/tRex_211022/result/ivt/" + key + "/pq/" + key + ".pq"
    outpq = "/share/trna/tyCooNNTest/trim12000IVT/" + key + ".pq"


    pqt = pq.read_table(ppathIvt,
                        columns=['read_id', 'score', 'reference_id', 'alnstart', 'cigar', 'otherhit', 'traceseq',
                                 'tbpath', 'trace', 'signal','mean_qscore'])

    dfp = pqt.to_pandas()
    cnt = 0
    wlen = 0
    datalist = []
    for idx, row in dfp.iterrows():

        readid = row[0]
        trna = key+"_ivt"
        trace = row[8]
        signal = row[9]
        mean_qscore = row[10]
        tbpath = row[7]

        if mean_qscore < 8:
            continue
        if len(tbpath) < 100:
            continue
        if len(signal) > 8000:
            continue

        start = tbpath[0]*10
        end = tbpath[-40]*10

        print(len(signal),start,end,len(tbpath))


        fig = plt.figure()
        plt.plot(signal)
        plt.axvline(x=start, color='b')
        plt.axvline(x=end, color='b')
        gm.add_figure(fig)

        fig = plt.figure()
        plt.plot(trace)
        gm.add_figure(fig)

        signal = signal[start:end]

        signal = signal[::-1]
        mediantoset  = np.median(signal)
        madv = mad(signal)
        normsig = (signal - mediantoset) / (madv * 5 )# med mad normalization instead
        formatsig = tn.binned(normsig, param.trimlen)
        zoformatsig = tn.zeroToOne(formatsig)
        datalist.append((readid,trna,zoformatsig))
        break
        cnt = cnt+1

    df = pd.DataFrame(datalist, columns=['read_id', 'trna', 'trimsignal'])
    df.to_parquet(outpq)

gm.save()