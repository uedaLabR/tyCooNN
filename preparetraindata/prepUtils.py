from tRex.TRexRead import Read
import numpy as np
ascii_order = '!\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
ascii_score_dict = {ascii_order[k]: k for k in range(len(ascii_order))}

class tRead:

    def __init__(self,tyRead):

        if tyRead is not None:

            self.read_id = tyRead.read_id
            signal = tyRead.signal
            trace = tyRead.trace
            self.adapter_signal = tyRead.adapter_signal
            self.signal =  tyRead.signal
            self.trace = tyRead.trace
            self.move = tyRead.move
            self.fastq = tyRead.fastq

            fastq_list = self.fastq.split('\n')
            self.sequence = fastq_list[1].replace('T', 'U')
            self.qscore = np.array([ascii_score_dict[symbol] for symbol in fastq_list[3]], dtype=np.int16)
            self.mean_qscore = sum(self.qscore) / len(self.qscore)
            # self.channel_number = channel_number
            # self.mux = mux
            # self.duration = duration
            # self.median_before = median_before
            # self.start_time = start_time

            self.normalized_signal = None

            self.mapping_results = []
            self.viterbi_results = []
            self.processid = 0

            # used for trim and normalize
            self.trimIdxbyHMM = tyRead.trimIdxbyHMM
            self.trimIdxbyMapping = tyRead.trimIdxbyMapping
            self.normalizeDelta = tyRead.normalizeDelta
            self.trimmedSignal = tyRead.trimmedSignal
            self.trimmedTrace = tyRead.trace
            self.normalizemed = tyRead.normalizemed
            self.inferencedtRNA = tyRead.inferencedtRNA


def changeClass(x):

    read_id = None
    signal = None
    trace = None
    move = None
    fastq = None
    channel_number = None
    mux = None
    duration = None
    median_before = None
    start_time = None
    tread = Read(read_id,signal,trace,move,fastq,channel_number,mux,duration,median_before,start_time)

    tread.read_id = x.read_id
    tread.signal = x.normSig
    tread.normSig = x.normSig
    tread.trace = x.trace
    tread.adapter_signal = x.adapter_signal
    tread.move = x.move
    tread.fastq = x.fastq

    fastq_list = x.fastq.split('\n')
    tread.sequence = fastq_list[1].replace('T', 'U')
    tread.qscore = np.array([ascii_score_dict[symbol] for symbol in fastq_list[3]], dtype=np.int16)
    tread.mean_qscore = sum(tread.qscore) / len(tread.qscore)

    # used for trim and normalize
    tread.trimIdxbyHMM = x.trimIdxbyHMM
    tread.trimIdxbyMapping = x.trimIdxbyMapping
    tread.normalizeDelta = x.normalizeDelta

    tread.normalizemed = x.normalizemed
    tread.inferencedtRNA = x.inferencedtRNA

    return tread

def convert(reads):

    return list(map(changeClass,reads))