import preparetraindata.inferanceAndMakeSegmentedPq as inference
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import tRex.TRexUtils as ut
from tRex.TRexReferenceHolder import ReferenceHolder
import tRex.viterbi.TRexViterbiOnTraceMinusScore2 as vb
import preparetraindata.prepUtils as pu
from preparetraindata.PairWiseAlgin import PairWiseAlgin as PairWiseAlgin
import tRex.io.TRexOutputUtils as outut
import os
import pandas as pd

VSCORE_THRES = 20
def testEvaluate(indirs,paramPath,postfix,outpath,configdir):



    mapping_option = ut.get_parameter('/home/ueda/project/tRex/test/ueda/pairwise_option.yaml')  # mapping parameter

    datalist = inference.evaluate(paramPath,indirs,configdir,outpath,postfix)
    VSCORE_THRES = 20
    param = ut.get_parameter('/home/ueda/project/tRex/test/ueda/settings.yaml')  # setting of file path and max core
    organism_name = 'EColi'  # EColi or Human
    rf = ReferenceHolder(unmod_path=param['reference_path'] + '/' + param['reference_name'][organism_name]['unmod'],
                         mod_path=param['reference_path'] + '/' + param['reference_name'][organism_name]['mod'],
                         five_prime_adapter_path=param['reference_path'] + '/' + param['reference_name'][organism_name][
                             'five_prime_adapter'],
                         three_prime_adapter_path=param['reference_path'] + '/' +
                                                  param['reference_name'][organism_name]['three_prime_adapter'])


    filtered_reads = pu.convert(datalist)
    genomeMapper = PairWiseAlgin(True)
    algined_reads = genomeMapper.genomeMap(rf, filtered_reads, mapping_option, param['max_core'])
    viterbi_reads = vb.flipplopViterbi(rf,algined_reads,param['max_core'],scorethres = VSCORE_THRES)
    #
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    #save to pq
    print('write to parquet file')
    pdata = []
    for read in viterbi_reads:
        pqread = outut.toPqRead(rf, read)
        pdata.append(pqread)
    df = pd.DataFrame(pdata, columns=['read_id', 'score','reference_id', 'alnstart','cigar','otherhit',
                                      'traceseq','tbpath','trace','signal','reference_name','refseq',
                                      'modseq','ismod','isprimer','mean_qscore'])

    u = df['reference_name'].unique()
    for rn in u:

        pqf = outpath+"/"+str(rn)+".pq"
        rn = "\""+ str(rn) + "\""
        qr = 'reference_name == '+rn
        dfselect = df.query(qr)
        dfselect.to_parquet(pqf)



paramPath = '/share/trna/tyCooNN/setting.yaml'

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

flist = []
flist.append(("_trub","/data/suzukilab/seqdata/basecall/200713_ecoli_TruB_total_v7/workspace"))
flist.append(("_smta","/data/suzukilab/seqdata/basecall/201026_ecoli_SmtA_total_LB_BW_sta_V7/workspace"))
flist.append(("_tapt","/data/suzukilab/seqdata/basecall/201027_ecoli_TapT_total_LB_BW_sta_V7/workspace"))
flist.append(("_tgt","/home/ueda/suzukif5/220412_ecoli_Tgt_total_LB_sta_v7/workspace"))
flist.append(("_trmj","/home/ueda/suzukif5/220412_ecoli_TrmJ_total_LB_sta_v7/workspace"))

flist.append(("_miab","/home/ueda/suzukif5/220413_ecoli_MiaB_total_LB_sta_v7/workspace"))
flist.append(("_mrluf","/home/ueda/suzukif5/220413_ecoli_RluF_total_LB_sta_v7/workspace"))
flist.append(("_trmb","/data/suzukilab/seqdata/basecall/201215_ecoli_TrmB_total_LB_BW_sta_V7/workspace"))

flist.append(("_mnmc","/home/ueda/suzukif5/220419_ecoli_MnmC_total_LB_sta_v7/workspace"))
flist.append(("_truA","/home/ueda/suzukif5/220419_ecoli_TruA_total_LB_sta_v7/workspace"))
flist.append(("_yfgb","/home/ueda/suzukif5/220419_ecoli_YfgB_total_LB_sta_v7/workspace"))

flist.append(("_dusa","/mnt/share/ueda/fast5/basecalled/221122_ecoli_dusA_total_LB_BW_sta_v7/workspace/no_sample/20221122_1518_MN34552_FAV44058_4e1b8eff/fast5"))
flist.append(("_dusb","/mnt/share/ueda/fast5/basecalled/221122_ecoli_dusB_total_LB_BW_sta_v7/workspace/no_sample/20221122_1519_MN35494_FAV44049_6a3b8c75/fast5"))
flist.append(("_dusc","/mnt/share/ueda/fast5/basecalled/221122_ecoli_dusC_total_LB_BW_sta_v7/workspace/no_sample/20221122_1520_MN35546_FAV46516_22587187/fast5"))


for postfix,indirs in flist:
    outpath = "/share/trna/tyCooNNTest/KO/"+postfix
    testEvaluate(indirs,paramPath,postfix,outpath,configdir)