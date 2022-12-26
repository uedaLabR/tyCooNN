import glob
import os
import training.GenaratePqForTrainning as pqg
import sys


def writeIOFile():

    path_w = "/share/trna/tyCooNNTest/inputsIVT.txt"
    fw = open(path_w, mode='w')

    #basecall_path = "/share/trna/testdata/ecolibasecalled/"

    basecall_path = "/data/suzukilab/seqdata/basecall/split/220820_ecoli_tRNA_IVT_batch3/fast5/species"
    files = glob.glob(basecall_path+"/*/")

    for f in files:

        dirlist = []
        fs = glob.glob(f+"/*/*/*/*/workspace")
        if len(fs)>0:
            dirlist = fs
        else:
            fs = glob.glob(f + "/*/*/*/workspace")
            dirlist = fs

        print(dirlist)
        basename = os.path.basename(os.path.dirname(f)).replace("ecoli_rcc_","")
        basename = basename.split("_")[1]
        #graphpath = "/share/trna/result/testcnntrex/"+basename+".pdf"
        pqpath = "/share/trna/tyCooNNTest/trim2400IVT/"+basename+".pq"
        lst = ",".join(dirlist)
        print(basename,lst,pqpath)
        fw.write(basename+"\t"+lst+"\t"+pqpath+" \n")

    fw.close()


#writeIOFile()

def writeIOFileChop():

    path_w = "/share/trna/tyCooNNTest/inputsIVT2.txt"
    fw = open(path_w, mode='w')

    #basecall_path = "/share/trna/testdata/ecolibasecalled/"

    basecall_path = "/data/suzukilab/seqdata/basecall/split/"
    # keys = ["Ala1B","Ala2","Arg2","Arg3","Arg4","Arg5","Asn","Asp","Cys","fMet1",
    #         "fMet2","Gln1","Gln2","Glu","Gly1","Gly2","Gly3","His","Ile1","Ile2",
    #         "Ile2v","Leu1","Leu1_P","Leu2","Leu3","Leu4","Leu5","Lys",
    #         "Met","Phe","Pro1","Pro2","Pro3","Sec","Ser1","Ser2","Ser3",
    #         "Ser5", "Thr1", "Thr2","Thr3","Thr4","Trp","Tyr1","Tyr2","Val1",
    #         "Val2A","Val2B"]
    keys = ["Ala2","Gly1"]

    for key in keys:

        dirlist = glob.glob(basecall_path+"/*/*/*/*"+key+".fast5")
        print(key,dirlist)
        basename = key+"_IVT"
        #graphpath = "/share/trna/result/testcnntrex/"+basename+".pdf"
        pqpath = "/share/trna/tyCooNNTest/trim2400IVT/"+basename+".pq"
        lst = ",".join(dirlist)
        print(basename,lst,pqpath)
        fw.write(basename+"\t"+lst+"\t"+pqpath+" \n")


    fw.close()

writeIOFileChop()

def genaratePq():

    #(paramPath, listOfIOPath, takeCount=12000)
    paramPath = '/share/trna/tyCooNN/setting.yaml'
    # inputs = "/share/trna/tyCooNNTest/inputs.txt"
    inputs = "/share/trna/tyCooNNTest/inputsIVT2.txt"
    pqg.generatePqForTrainingAll(paramPath,inputs,takeCount=2400)

genaratePq()

