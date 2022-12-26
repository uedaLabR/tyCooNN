import pyarrow.parquet as pq
import mappy as mp
import pysam


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


import glob
def getFiles(path, chrom, strand, start, end):

    sortedfile = []

    l = glob.glob(path+"/*/*/*/*.parquet")
    for fileNamePath in l:

        chromFromPath = getChrom(fileNamePath)
        strandP = getStrand(fileNamePath) == 1
        sortKey = getBinkey(fileNamePath)
        #
        keyCheck = _keyCheck(start, end, sortKey)

        if (chromFromPath == chrom) and (strand == strandP) and keyCheck:

            if checkMeta(fileNamePath, start, end):
                sortedfile.append(fileNamePath)

    return sortedfile

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



    def __init__(self,dir):

        #files = self.getFiles(path,chrom,strand,findexs)
        data = None
        sortedfile = getFiles(dir)
        data = None
        for file in sortedfile:

            dataadd = pq.read_table(file, columns=['trna','r_st', 'r_en','q_st','q_en','cigar','traceintervals','signal']).to_pandas()
            if data is None:
                data = dataadd
            else:
                data = pd.concat([data, dataadd])

        #

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


    def getRelativePos(self, start, end, cigar, pos):

        if self.strand == True:
            return self.getRelativePosP(start, end, cigar, pos)
        else:
            return self.getRelativePosN(start, end, cigar, pos)

    def getRelativePosP(self, _start, end, cigar, pos):

        # tp = (strand,start,end,cigar,pos,traceintervalLen)
        rel0 = pos - _start
        rel = self.correctCigar(rel0, cigar)
        start = rel
        startmargin = 8
        if start < startmargin:
            # do not use lower end
            return None

        rel = pos - _start + 6
        end = self.correctCigar(rel, cigar)
        if end < 0: # intron
            end = start+6
        return start, end

    def getRelativePosN(self, start, end, cigar, pos):

        # tp = (strand,start,end,cigar,pos,traceintervalLen)
        margin = 1
        rel0 = end - pos - margin
        rel = self.correctCigar(rel0, cigar)
        start = rel
        startmargin = 8
        if start < startmargin:
            # do not use lower end
            return None

        rel = end - pos + 6 - margin
        end = self.correctCigar(rel, cigar)
        if end < 0: # intron
            end = start+6
        # print("start-end",start, end)
        return start, end

    def calcStartEnd(self,start,end,cigar,pos):

        rp = self.getRelativePos(start,end,cigar,pos)
        if rp is None:
            return None
        # print(rp)
        #print("offset", offset)
        relativeStart, relativeEnd = rp
        if abs(relativeStart-relativeEnd) != 6:
            return None
        return relativeStart,relativeEnd



    def getOneRow(self,row, pos):

        start = row['r_st']
        end = row['r_en']
        q_start = row['q_st']
        q_end = row['q_en']

        cigar = row['cigar']
        traceboundary = row['traceintervals']
        signal = row['signal']
        UNIT = 10
        UNITLENGTH = 1024

        sted = self.calcStartEnd(start,end,cigar,pos)
        # print(sted,start,end,cigar,pos)
        if sted is None:
            return None
        relativeStart,relativeEnd = sted
        signal_start = traceboundary[relativeStart]*UNIT
        signal_end = traceboundary[relativeEnd]*UNIT

        subsignal = signal[signal_start:signal_end]

        if len(subsignal) == 0:
            return None

        info = (start, end, q_start, q_end, cigar)

        binsignal = self.binSignal(subsignal,UNITLENGTH)
        if binsignal is None:
            return None

        return  binsignal, info

    def getRowData(self, pos, takecnt=-1):

        reloadUnit = 100
        if self.bufData is None or abs(pos-self.start) % reloadUnit:
            self.load(pos)

        initfilterdata = []
        infos = []
        for index, row in self.bufData.iterrows():

            ret = self.getOneRow(row, pos)
            if ret is not None:
                # print(ret.shape)
                v,i = ret
                initfilterdata.append(v)
                infos.append(i)
            if len(initfilterdata) == takecnt:
                break

        return initfilterdata,infos



