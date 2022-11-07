import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import inference.VaridateActialACC as varidate
import utils.labelMatrixUtil as labelMatrixUtil


def testTrain():

    dirpath = "/share/trna/tyCooNNTest/trim12000/"
    outdir = "/share/trna/testbasecalled/test"
    sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq.csv"
    modkind = "/share/trna/testbasecalled/modkind.csv"

    labeldic = labelMatrixUtil.getLabelMatrixDic(sequencemaxrix,modkind)
    modlist = labelMatrixUtil.toNameList(modkind)

    epoch = 100
    weightpath = "/share/trna/testbasecalled/test/learent_weight.h5"
    varidate.varidate(dirpath,outdir,labeldic,modlist,weightpath)


    # epoch = 10
    #traning.train(input, outdir, epoch,data_argument =3)

testTrain()

