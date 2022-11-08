import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import training.TrainBaseCall as traning
import utils.labelMatrixUtil as labelMatrixUtil


def testTrain():

    # input = "/share/trna/tyCooNNTest/trim12000/"
    input = "/share/trna/tyCooNNTest/trim12000"
    input2 = "/share/trna/tyCooNNTest/KO/selected"
    outdir = "/share/trna/testbasecalled/testko3"


    # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq.csv"
    # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVT.csv"
    sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_KO.csv"
    modkind = "/share/trna/testbasecalled/modkind.csv"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    labeldic = labelMatrixUtil.getLabelMatrixDic(sequencemaxrix,modkind)

    epoch = 100
    traning.train(input,input2, outdir,labeldic, epoch)

    # epoch = 10
    #traning.train(input, outdir, epoch,data_argument =3)

testTrain()

