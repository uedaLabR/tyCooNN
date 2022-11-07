import glob
from pyarrow import parquet as pq
from utils.GraphManager import GraphManager
import matplotlib.pyplot as plt

input = "/share/trna/tyCooNNTest/trim12000/"
input2 = "/share/trna/tyCooNNTest/trim12000IVT"

# input = "/share/bhaskar/tRex_211022/result/ivt/Ala2/pq"
# input2 = None

def train(dirpath,dirpath2):

    gm = GraphManager("/share/trna/tyCooNNTest/test.pdf")

    print(dirpath)
    fs = glob.glob(dirpath + "/*.pq*")
    #fs = fs[0:3] #for debug
    if dirpath2 is not None:
        fs2 = glob.glob(dirpath2 + "/*.pq*")
        fs.extend(fs2)

    trnas = []

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    wlen = 0
    fcnt = 0
    for f in fs:

        postfix =""
        # if dirpath2 in f:
        #     postfix = "_ivt"
        print(f,postfix)


        pqt = pq.read_table(f,
                            columns=['trimsignal'])

        dfp = pqt.to_pandas()
        cnt = 0
        wlen = 0
        signals = []
        for idx, row in dfp.iterrows():


            signal = row[0]
            print(signal)
            fig = plt.figure()
            plt.plot(signal)
            gm.add_figure(fig)
            # if cnt == 30:
            #     break
            break
            cnt+=1
        print(cnt)

    gm.save()

train(input,input2)

