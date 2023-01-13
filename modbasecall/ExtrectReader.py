import pyarrow.parquet as pq
import mappy as mp
import pysam
import pandas as pd


def getChrom(f):

    return f.split("/")[-3].replace("chrom=","")

def getStrand(f):

    return f.split("/")[-2].replace("strand=","") == "0"

def getFiles(self,path,chrom,strand,findexs):

    sortedfile = []
    for root, subFolder, files in os.walk(path):
        for item in files:
            if item.endswith(".parquet"):
                fileNamePath = str(os.path.join(root, item))
                chromFromPath = getChrom(fileNamePath)
                strandP = getStrand(fileNamePath)

                if (chromFromPath == chrom) and (strand == strandP):
                    sortedfile.append(fileNamePath)

    sortedfile = sorted(sortedfile)
    ret = []
    n = 0
    for f in sortedfile:
        if n in findexs:
            ret.append(f)
        n+=1

    return ret

import os
import pandas as pd
import numpy as np
from scipy import ndimage as ndi
import statistics


def intervalToAbsolute(intervals):
    ret = []
    cnt = 0
    sum = 0
    for n in intervals:

        if cnt == 0:
            ret.append(0)
            sum = sum + n
            cnt += 1
            continue
        # else
        ret.append(sum)
        sum = sum + n

    ret.append(sum)
    return np.array(ret)

def _keyCheck(start, end, sortKey):
    sortkey = int(sortKey)
    binsize = 100000
    flg = False
    k1 = start // binsize
    k2 = end // binsize
    if k1 == sortkey or k2 == sortkey or k1 - 1 == sortkey:
        flg = True
    return flg


def checkMeta(fileNamePath, start, end):
    parquet_file = pq.ParquetFile(fileNamePath)
    posmin = parquet_file.metadata.row_group(0).column(3).statistics.min
    posmax = parquet_file.metadata.row_group(0).column(4).statistics.max

    return posmax > start and end > posmin  # intersect


def getChrom(f):
    return f.split("/")[-4].replace("chrom=", "")


def getStrand(f):
    return f.split("/")[-3].replace("strand=", "")

def getBinkey(f):
    return f.split("/")[-2].replace("sortkey=", "")



import scipy.signal as scisignal
from scipy.interpolate import interp1d
class RowReader:

    def getChrom(self,f):

        return f.split("/")[-2].replace("chrom=", "")

    def getStrand(self,f):

        return f.split("/")[-1].replace("strand=", "")

    def getDepth(self, pos):

        # extract parquet file contain reads in this region
        # query = 'start <= ' + str(pos) + ' & end >= ' + str(pos) + ' & chr == "' + self.chr + "'"
        if self.indexdata is None or pos % 100 == 0:

            sortedfile = self.getFiles(self.path, self.chrom, self.strand, self.findexs, all=True)
            indexdata = None
            for filepath in sortedfile:

                if indexdata is None:
                    indexdata = pq.read_table(filepath, columns=['r_st', 'r_en']).to_pandas()

                else:
                    dataadd = pq.read_table(filepath, columns=['r_st', 'r_en']).to_pandas()
                    indexdata = pd.concat([indexdata, dataadd])

            self.indexdata = indexdata

        depth = self.indexdata.query('r_st <=' + str(pos - 10) + ' & r_en >=' + str(pos + 10))['r_st'].count()
        return depth



    def __init__(self,file):

        #files = self.getFiles(path,chrom,strand,findexs)
        #
        # print(file)
        self.df = pd.read_parquet(file, columns=['read_id', 'score', 'reference_id', 'alnstart', 'cigar', 'otherhit',
                                          'traceseq', 'tbpath', 'trace', 'signal', 'reference_name', 'refseq',
                                          'modseq', 'ismod', 'isprimer', 'mean_qscore'])



    import pysam
    def correctCigar(self,targetPos,cigar):

        a = pysam.AlignedSegment()
        a.cigarstring = cigar
        refpos = 0
        relpos = 0
        cidx = 0
        for cigaroprator, cigarlen in a.cigar:

            if cigaroprator == 3 and cidx > 0: #N

                if refpos + cigarlen > targetPos:
                    return -1

                refpos = refpos + cigarlen

            elif cigaroprator == 0 or cigaroprator == 4:  # match or S softclip was not correted so treat as M

                if refpos + cigarlen > targetPos:
                    return relpos + (targetPos - refpos)

                relpos = relpos + cigarlen
                refpos = refpos + cigarlen

            elif cigaroprator == 2:  # Del

                refpos = refpos + cigarlen

            elif cigaroprator == 1 :  # Ins

                if relpos == 0:
                    if targetPos <= cigarlen:
                        return 0

                relpos = relpos + cigarlen

        cidx+=1
        return 0






    def getOneRow(self,row, start,end):

        a_start = row['alnstart']
        traceboundary =  row['tbpath']

        cigar = row['cigar']
        signal = row['signal']
        primerlen = 4

        UNIT = 10
        UNITLENGTH = 1024


        start = start - a_start + primerlen
        end = end - a_start + primerlen
        # print("a_start",a_start)

        start = self.correctCigar(start, cigar)
        end = self.correctCigar(end, cigar)

        if start < 0 or end < 0:
            return None
        if start > len(traceboundary) or end > len(traceboundary):
            return None
        if signal is None:
            return None

        signal_start = traceboundary[start]*UNIT
        signal_end = traceboundary[end]*UNIT
        subsignal = signal[signal_start:signal_end]

        binsignal = self.binSignal(subsignal,UNITLENGTH)
        return  binsignal


    def downsample(self,array, npts):

        interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
        downsampled = interpolated(np.linspace(0, len(array), npts))
        # downsampled = scisignal.resample(array, npts)
        return downsampled

    def binSignal(self,trimsignal, trimlength, mode=1):

        if len(trimsignal) == trimlength:
            return trimsignal  # not very likely to happen

        if len(trimsignal) > trimlength:
            # trim from first
            return self.downsample(trimsignal, trimlength)

        else:
            #
            trimsignal = trimsignal.astype(np.float32)
            siglen = len(trimsignal)
            left = trimlength - siglen
            lefthalf = left // 2

            leftlen = trimlength - siglen - lefthalf
            ret = np.concatenate([np.zeros(lefthalf), trimsignal, np.zeros(leftlen)])

            return ret

    def getRowData(self, start,end, takecnt=-1):

        initfilterdata = []
        for index, row in self.df.iterrows():

            ret = self.getOneRow(row, start,end)
            if ret is not None:
                # print(ret.shape)
                initfilterdata.append(ret)
            if len(initfilterdata) == takecnt:
                break

        return initfilterdata



