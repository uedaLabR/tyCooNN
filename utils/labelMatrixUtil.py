import numpy as np

def toNameList(modkind):

    ret = []
    with open(modkind, encoding='utf-8') as f:

        lines = f.readlines()
        lines = list(map(lambda l: l.rstrip("\n"), lines))

        ret.append("")
        for line in lines:
            ret.append(line.lower())

    return ret

def toLabelM(nl,adata):

    da = np.zeros((99,34))
    # print(len(adata))
    for n in range(len(adata)):
        s = adata[n]
        s = s.lower()
        idx = 0
        if s in nl:
            idx = nl.index(s)
            # print(idx,s,nl)
        da[n][idx] = 1.0

    return da

def toLabelMNum(nl,adata,num):

    s = 0
    e = 10
    for m in range(num):
        s+=5
        e+=5


    da = np.zeros((10,34))
    # print(len(adata))
    for n in range(len(adata)):

        if n>=s and n<=e:
            st = adata[n]
            st = st.lower()
            idx = 0
            if st in nl:
                idx = nl.index(st)
                # print(idx,s,nl)
            da[n-s][idx] = 1.0

    return da

def getLabelMatrixDic(sequencemaxrix,modkind):

    nl = toNameList(modkind)
    d = {}
    with open(sequencemaxrix, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            aline = line.split(',')
            alabel = aline[0].lower()
            adata = aline[1:]

            lm = toLabelM(nl,adata)

            print(alabel, lm)
            d[alabel] = lm

    return d

def getLabelMatrixDicNum(sequencemaxrix,modkind,num):

    nl = toNameList(modkind)
    d = {}
    with open(sequencemaxrix, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            aline = line.split(',')
            alabel = aline[0].lower()
            adata = aline[1:]

            lm = toLabelMNum(nl,adata)

            print(alabel, lm)
            d[alabel] = lm

    return d

def getNameList(sequencemaxrix):

    nl = []
    with open(sequencemaxrix, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            aline = line.split(',')
            alabel = aline[0].lower()
            nl.append(alabel)
    return nl

def getMaxCandidate(scorematrix,labeldict):

    maxscore = 0
    maxkey = ""
    for k in labeldict:

        matrixB = labeldict[k]
        score = np.vdot(scorematrix, matrixB)
        if score > maxscore:

            maxscore = score
            maxkey = k

    return maxkey

