import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import inference.VaridateActialACC as varidate
import utils.labelMatrixUtil as labelMatrixUtil


def testTrain():

    # input = "/share/trna/tyCooNNTest/trim12000/"
    # input2 = "/share/trna/tyCooNNTest/trim12000IVT/"

    input = "/share/trna/tyCooNNTest/trim2400"
    input2 = "/share/trna/tyCooNNTest/KO/selected"
    input3 = "/share/trna/tyCooNNTest/trim2400IVT"

    inputs = [input,input2,input3]

    outdir = "/share/trna/testbasecalled/testkoivt2400"
   # sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq.csv"
    #sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVT.csv"
    sequencemaxrix = "/share/trna/testbasecalled/ecoliModseq_IVTKO.csv"
    modkind = "/share/trna/testbasecalled/modkind.csv"

    labeldic = labelMatrixUtil.getLabelMatrixDic(sequencemaxrix,modkind)
    modlist = labelMatrixUtil.toNameList(modkind)
    samplenames = labelMatrixUtil.getNameList(sequencemaxrix)
    print(samplenames)
    epoch = 100
   # weightpath = "/share/trna/testbasecalled/test/learent_weight.h5"
   # weightpath = "/share/trna/testbasecalled/testivt/learent_weight.h5"
    weightpath = "/share/trna/testbasecalled/testkoivt2400/learent_weight.h5"
    varidate.varidate(inputs, outdir,labeldic,modlist,samplenames,weightpath)


    # epoch = 10
    #traning.train(input, outdir, epoch,data_argument =3)

import os
import tensorflow as tf
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
with tf.device('/CPU:0'):
    testTrain()

