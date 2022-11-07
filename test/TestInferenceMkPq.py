import inference.InferenceAndMakePq as inference

def testEvaluate():


    # indirs = "//data/suzukilab/seqdata/basecall/200713_ecoli_TruB_total_v7/workspace"
    #indirs = "/data/suzukilab/seqdata/basecall/201026_ecoli_SmtA_total_LB_BW_sta_V7/workspace"
    #indirs = "/data/suzukilab/seqdata/basecall/201027_ecoli_TapT_total_LB_BW_sta_V7/workspace"
    # indirs = "/home/ueda/suzukif5/220412_ecoli_Tgt_total_LB_sta_v7/workspace"
    # indirs = "/home/ueda/suzukif5/220412_ecoli_TrmJ_total_LB_sta_v7/workspace"
    #indirs = "/home/ueda/suzukif5/220413_ecoli_MiaB_total_LB_sta_v7/workspace"
    #indirs = "/home/ueda/suzukif5/220413_ecoli_RluF_total_LB_sta_v7/workspace"
    #indirs = "/data/suzukilab/seqdata/basecall/201215_ecoli_TrmB_total_LB_BW_sta_V7/workspace"
    #indirs = "/home/ueda/suzukif5/220419_ecoli_MnmC_total_LB_sta_v7/workspace"
    #indirs = "/home/ueda/suzukif5/220419_ecoli_TruA_total_LB_sta_v7/workspace"
    #indirs = "/home/ueda/suzukif5/220419_ecoli_YfgB_total_LB_sta_v7/workspace"
    indirs = "/data/suzukilab/seqdata/basecall/201026_ecoli_SmtA_total_LB_BW_sta_V7/workspace"

    configdir = "/share/trna/tyCooNNTest/testout"

    postfix = "_smta"
    outpath = "/share/trna/tyCooNNTest/KO/"+postfix

    paramPath = '/share/trna/tyCooNN/setting.yaml'
    inference.evaluate(paramPath,indirs,configdir,outpath,postfix)


testEvaluate()