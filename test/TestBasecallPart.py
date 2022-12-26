import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import training.TrainBaseCall as traning
import utils.labelMatrixUtil as labelMatrixUtil


def testTrain():

    # input = "/share/trna/tyCooNNTest/trim12000/"
    #input = "/share/trna/tyCooNNTest/trim12000"
    input = "/share/trna/tyCooNNTest/trim2400"
    input2 = "/share/trna/tyCooNNTest/trim2400IVT"

    inputs = [input,input2]
    outdir = "/share/trna/testbasecalled/testpartialbc"


    # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq.csv"
    # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVT.csv"
    # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_KO.csv"
    sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVTKO.csv"
    modkind = "/share/trna/testbasecalled/modkind.csv"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    labeldic = labelMatrixUtil.getLabelMatrixDic(sequencemaxrix,modkind)

    # epoch = 100
    # traning.train(inputs, outdir,labeldic, epoch)

    epoch = 100
    traning.train(inputs, outdir,labeldic, epoch,data_argument = 2)

# import tensorflow as tf
# with tf.device('/CPU:0'):
#     testTrain()

testTrain()

