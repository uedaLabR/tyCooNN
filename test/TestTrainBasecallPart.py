import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import training.TrainBaseCallPart as traning
import utils.labelMatrixUtil as labelMatrixUtil


def testTrain():


    input = "/share/trna/tyCooNNTest/trim2400"
    input2 = "/share/trna/tyCooNNTest/KO/"
    input3 = "/share/trna/tyCooNNTest/trim2400IVT"

    inputs = [input,input3,input2]
    outdir = "/share/trna/testbasecalled/testPartial/"

    sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVTKO.csv"
    modkind = "/share/trna/testbasecalled/modkind.csv"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    labeldic = labelMatrixUtil.getLabelMatrixDic(sequencemaxrix,modkind)


    epoch = 100
    portions = 8
    for portion in range(portions):

        traning.train(inputs, outdir,labeldic,portion, epoch,data_argument = 2)


testTrain()

