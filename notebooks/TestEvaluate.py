import glob
import os
import training.GenaratePqForTrainning as pqg
import sys
import training.Evaluate as evaluate

def testEvaluate():

    input = "/share/trna/tyCooNNTest/trim12000/"
    outdir = "/share/trna/tyCooNNTest/testout"
    csvout = "/share/trna/tyCooNNTest/test.csv"
    evaluate.evaluate(input, outdir, csvout)


testEvaluate()