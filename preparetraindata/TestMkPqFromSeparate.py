import preparetraindata.inferanceAndMakeSegmentedPq as inference
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import tRex.TRexUtils as ut
import utils.tyUtils as tu
from tRex.TRexReferenceHolder import ReferenceHolder
import tRex.viterbi.TRexViterbiOnTraceMinusScore2 as vb
import preparetraindata.prepUtils as pu
from preparetraindata.PairWiseAlgin import PairWiseAlgin as PairWiseAlgin
import tRex.io.TRexOutputUtils as outut
import statistics

def getKey(species):
    margeMap = {"ala1": "Ala1B", "fmet": "fMet1", "tyr": "Tyr1"}
    if species in margeMap:
        return margeMap[species]
    return species


def getHighAGPeak(sigfrom3p):

    m = 0
    medmax = 0
    for n in range(5):
        firstsigfrom3p = sigfrom3p[m:m + 500]
        if len(firstsigfrom3p) == 0:
            break
        med = statistics.median(firstsigfrom3p)
        if medmax < med:
            medmax = med
        m = m + 100
    return medmax

def normalize(read, param):

    sigfrom3p = read.signal
    adaptor1med = 0
    if len(read.adapter_signal) > 0:
        adaptor1med = statistics.median(read.adapter_signal)

    # take median of first 50-550
    adaptor2med = getHighAGPeak(sigfrom3p)
    diffadop = adaptor2med - adaptor1med
    hmmIndex = 0
    normsig = None
    mediantoset = 0
    if diffadop > 20:

        diffthery = param.adap2thery - param.adap1thery
        step = diffadop / diffthery
        mediantoset = adaptor1med + (step * (param.meantoSet - param.adap1thery))
        normsig = (sigfrom3p - mediantoset) / diffadop
        read.normSig = normsig

VSCORE_THRES = 20
def testEvaluate(indirs,tRNALabel, paramPath, postfix, outpq, configdir,takeCount=2400):




    VSCORE_THRES = 20

    organism_name = 'EColi'  # EColi or Human
    f5s =indirs.split(",")
    filtered_reads = ut.get_fast5_reads_from_list_mt(f5s,20)
    # filtered_reads = []
    # for ff in f5s:
    #     _reads = ut.get_fast5_reads(ff, 20)
    #     filtered_reads.extend(_reads)

    filtered_reads = filtered_reads[0:10000]

    typaram = tu.get_parameter(paramPath)
    param = ut.get_parameter('/home/ueda/project/tRex/test/ueda/settings.yaml')
    mapping_option = ut.get_parameter('/home/ueda/project/tRex/test/ueda/pairwise_option.yaml')

    tRNALabel = getKey(tRNALabel)

    reads = []
    for read in filtered_reads:

        normalize(read,typaram)
        if read.normSig is not None:
            read.inferencedtRNA = tRNALabel
            reads.append(read)


    rf = ReferenceHolder(unmod_path=param['reference_path'] + '/' + param['reference_name'][organism_name]['unmod'],
                         mod_path=param['reference_path'] + '/' + param['reference_name'][organism_name]['mod'],
                         five_prime_adapter_path=param['reference_path'] + '/' + param['reference_name'][organism_name][
                             'five_prime_adapter'],
                         three_prime_adapter_path=param['reference_path'] + '/' +
                                                  param['reference_name'][organism_name]['three_prime_adapter'])

    # print(rf)
    genomeMapper = PairWiseAlgin(False)
    algined_reads = genomeMapper.genomeMap(rf, reads, mapping_option, param['max_core'])
    viterbi_reads = vb.flipplopViterbi(rf,algined_reads,param['max_core'],scorethres = VSCORE_THRES)
    #
    #save to pq
    outut.writePq(rf, viterbi_reads, outpq)


paramPath = '/share/trna/tyCooNN/setting.yaml'
postfix = "_smta"
outpath = "/mnt/share/ueda/tyCooNNTest/IVT/"+postfix
configdir = "/share/trna/tyCooNNTest/testout"

# indirs = "//data/suzukilab/seqdata/basecall/200713_ecoli_TruB_total_v7/workspace"
# indirs = "/data/suzukilab/seqdata/basecall/201026_ecoli_SmtA_total_LB_BW_sta_V7/workspace"
# indirs = "/data/suzukilab/seqdata/basecall/201027_ecoli_TapT_total_LB_BW_sta_V7/workspace"
# indirs = "/home/ueda/suzukif5/220412_ecoli_Tgt_total_LB_sta_v7/workspace"
# indirs = "/home/ueda/suzukif5/220412_ecoli_TrmJ_total_LB_sta_v7/workspace"
# indirs = "/home/ueda/suzukif5/220413_ecoli_MiaB_total_LB_sta_v7/workspace"
# indirs = "/home/ueda/suzukif5/220413_ecoli_RluF_total_LB_sta_v7/workspace"
# indirs = "/data/suzukilab/seqdata/basecall/201215_ecoli_TrmB_total_LB_BW_sta_V7/workspace"
# indirs = "/home/ueda/suzukif5/220419_ecoli_MnmC_total_LB_sta_v7/workspace"
# indirs = "/home/ueda/suzukif5/220419_ecoli_TruA_total_LB_sta_v7/workspace"
# indirs = "/home/ueda/suzukif5/220419_ecoli_YfgB_total_LB_sta_v7/workspace"
indirs = "/data/suzukilab/seqdata/basecall/201026_ecoli_SmtA_total_LB_BW_sta_V7/workspace"



import glob
import os
import training.GenaratePqForTrainning as pqg


import sys
folder_path = "/home/ueda/project/tRex/test/ueda/"
sys.path.insert(0, folder_path)

# def writeIOFile():
#
#     path_w = "/share/trna/tyCooNNTest/inputs.txt"
#     fw = open(path_w, mode='w')
#
#     basecall_path = "/share/trna/testdata/ecolibasecalled/"
#     files = glob.glob(basecall_path+"/*/")
#
#     for f in files:
#
#         dirlist = []
#         fs = glob.glob(f+"/*/*/*/*/workspace")
#         if len(fs)>0:
#             dirlist = fs
#         else:
#             fs = glob.glob(f + "/*/*/*/workspace")
#             dirlist = fs
#
#         basename = os.path.basename(os.path.dirname(f)).replace("ecoli_rcc_","")
#         basename = basename.split("_")[1]
#         #graphpath = "/share/trna/result/testcnntrex/"+basename+".pdf"
#         pqpath = "/share/trna/tyCooNNTest/trim2400/"+basename+".pq"
#         lst = ",".join(dirlist)
#         print(basename,lst,pqpath)
#         fw.write(basename+"\t"+lst+"\t"+pqpath+" \n")
#
#     fw.close()


# writeIOFile()

import csv
def genaratePq():

    #(paramPath, listOfIOPath, takeCount=12000)
    paramPath = '/share/trna/tyCooNN/setting.yaml'
    #inputs = "/share/trna/tyCooNNTest/inputs.txt"
    inputs = "/share/trna/tyCooNNTest/inputsIVT.txt"

    f = open(inputs,'r')
    tsv = csv.reader(f, delimiter='\t')

    # tRNAlabel indir   outpq
    for row in tsv:

        tRNALabel, indirs, outpq = row[0],row[1],row[2]
        outpq = outpq.strip(" ")
        outpq = outpq.strip("\t")
        print("doing..",tRNALabel,outpq)
        print(indirs)
        tRNALabel = tRNALabel.replace("_IVT","")
        print("tRNALabel",tRNALabel)
        #if tRNALabel == "ala1" or tRNALabel == "tyr" or tRNALabel == "fmet" or tRNALabel == "fmet":
        testEvaluate(indirs, tRNALabel,paramPath, postfix, outpq, configdir, takeCount=2400)



genaratePq()

