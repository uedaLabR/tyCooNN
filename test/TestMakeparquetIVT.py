import glob
import os
import training.GenaratePqForTrainning as pqg
import sys


def writeIOFile():

    path_w = "/share/trna/tyCooNNTest/inputsIVT.txt"
    fw = open(path_w, mode='w')

    basecall_path = "/share/trna/testdata/ecolibasecalled/"
    files = glob.glob(basecall_path+"/*/")

    for f in files:

        dirlist = []
        fs = glob.glob(f+"/*/*/*/*/workspace")
        if len(fs)>0:
            dirlist = fs
        else:
            fs = glob.glob(f + "/*/*/*/workspace")
            dirlist = fs

        basename = os.path.basename(os.path.dirname(f)).replace("ecoli_rcc_","")
        basename = basename.split("_")[1]
        #graphpath = "/share/trna/result/testcnntrex/"+basename+".pdf"
        pqpath = "/share/trna/tyCooNNTest/trim12000/"+basename+".pq"
        lst = ",".join(dirlist)
        print(basename,lst,pqpath)
        fw.write(basename+"\t"+lst+"\t"+pqpath+" \n")

    fw.close()


writeIOFile()



def genaratePq():

    #(paramPath, listOfIOPath, takeCount=12000)
    paramPath = '/share/trna/tyCooNN/setting.yaml'
    # inputs = "/share/trna/tyCooNNTest/inputs.txt"
    inputs = "/share/trna/tyCooNNTest/inputs.txt"
    pqg.generatePqForTrainingAll(paramPath,inputs,takeCount=1200)

genaratePq()

