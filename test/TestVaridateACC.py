import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import inference.VaridateActialACC as varidate
import utils.labelMatrixUtil as labelMatrixUtil


def testTrain():

    # input = "/share/trna/tyCooNNTest/trim12000/"
    # input2 = "/share/trna/tyCooNNTest/trim12000IVT/"

    input = "/share/trna/tyCooNNTest/trim12000"
    input2 = "/share/trna/tyCooNNTest/KO/selected"

    outdir = "/share/trna/testbasecalled/testko2"
   # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq.csv"
    #sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVT.csv"
    sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_KO.csv"
    modkind = "/share/trna/testbasecalled/modkind.csv"

    labeldic = labelMatrixUtil.getLabelMatrixDic(sequencemaxrix,modkind)
    modlist = labelMatrixUtil.toNameList(modkind)
    samplenames = labelMatrixUtil.getNameList(sequencemaxrix)
    print(samplenames)
    epoch = 100
   # weightpath = "/share/trna/testbasecalled/test/learent_weight.h5"
   # weightpath = "/share/trna/testbasecalled/testivt/learent_weight.h5"
    weightpath = "/share/trna/testbasecalled/testko2/learent_weight.h5"
    varidate.varidate(input,input2, outdir,labeldic,modlist,samplenames,weightpath)


    # epoch = 10
    #traning.train(input, outdir, epoch,data_argument =3)

testTrain()

