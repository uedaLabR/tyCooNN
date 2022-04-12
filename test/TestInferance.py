import inference.Inference as inference

def testEvaluate():

    #indirs = "/share/trna/testdata/210213_ecoli_WT_total_LB_BW_log_1_V7/1/20210213_1049_MN23375_FAO46570_a665fb8a/fast5/workspace"
    #indirs = "/share/trna/testdata/210213_ecoli_WT_total_LB_BW_log_2_V7/1/20210213_1050_MN29778_FAP04280_7587a04a/fast5/workspace"
    #indirs ="/share/trna/testdata/210213_ecoli_WT_total_LB_BW_sta_1_V7/1/20210213_0432_MN23375_FAP27902_bfc1e988/fast5/workspace"
    #indirs = "/share/trna/testdata/201026_ecoli_SmtA_total_LB_BW_sta_V7/1/20201026_0916_MN29778_FAO47177_7299feae/fast5/workspace,/share/trna/testdata/201026_ecoli_SmtA_total_LB_BW_sta_V7/2/20201026_0927_MN29778_FAO47177_f56e19af/fast5/workspace"
    indirs = "/share/trna/testdata/201026_ecoli_WT_mcmo5U_total_LB_BW_sta_V7_workspace/1/20201026_0914_MN23375_FAO49608_da8b7f1c/fast5/workspace,/share/trna/testdata/201026_ecoli_WT_mcmo5U_total_LB_BW_sta_V7_workspace/2/20201026_0925_MN23375_FAO49608_c7f935e1/fast5/workspace"

    configdir = "/share/trna/tyCooNNTest/testout"

    outpath = "/share/trna/tyCooNNTest/WT_mcmo5U_total_LB_BW"

    fasta = "/share/trna/tyCooNNTest/ecolitRNA_full.fa"
    paramPath = '/share/trna/tyCooNN/setting.yaml'

    inference.evaluate(paramPath,indirs,configdir,outpath,fasta,"S")


testEvaluate()