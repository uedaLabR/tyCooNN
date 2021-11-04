import inference.Inference as inference

def testEvaluate():

    indirs = "/share/trna/testdata/210213_ecoli_WT_total_LB_BW_log_1_V7/1/20210213_1049_MN23375_FAO46570_a665fb8a/fast5/workspace"
    configdir = "/share/trna/tyCooNNTest/testout"
    outpath = "/share/trna/tyCooNNTest/testout_total"
    fasta = "/share/trna/tyCooNNTest/ecolitRNA_full.fa"
    paramPath = '/share/trna/tyCooNN/setting.yaml'

    inference.evaluate(paramPath,indirs,configdir,outpath,fasta)


testEvaluate()