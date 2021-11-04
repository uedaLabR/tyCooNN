import numpy as np

class Counter():


    def __init__(self, trnas):

        tlen = len(trnas)

        self.filterFlgCnt = np.zeros(9)
        self.passfilterCnt = np.zeros(tlen)
        self.allCnt = np.zeros(tlen)

    def sumup(self,counter):

        self.filterFlgCnt = np.add(self.filterFlgCnt,counter.filterFlgCnt)
        self.passfilterCnt = np.add(self.passfilterCnt, counter.passfilterCnt)
        self.allCnt = np.add(self.allCnt, counter.allCnt)

    def inc(self,minicnt):

        self.filterFlgCnt[minicnt.filterFlg] = self.filterFlgCnt[minicnt.filterFlg]+1
        if minicnt.filterFlg ==0 and minicnt.trimSuccess and minicnt.maxval > 0.75:

            self.passfilterCnt[minicnt.tRNAIdx] =  self.passfilterCnt[minicnt.tRNAIdx]+1

        if  minicnt.trimSuccess:

            self.allCnt[minicnt.tRNAIdx] = self.allCnt[minicnt.tRNAIdx] + 1




class MiniCounter():

    def __init__(self,filterflg,trimSuccess):

        self.filterFlg  = filterflg
        self.trimSuccess = trimSuccess
        self.tRNA = None
        self.tRNAIdx = 0
        self.maxval = 0.0

    def addInference(self, tRNA,tRNAIdx,maxval):

        self.tRNA = tRNA
        self.tRNAIdx = tRNAIdx
        self.maxval = maxval