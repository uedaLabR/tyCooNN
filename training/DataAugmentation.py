import pandas as pd
import tensorflow as tf
from tensorflow import keras
import nnmodels.CNNWavenet as cnnwavenet
from numba import jit
import multiprocessing
import csv
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import utils.tyUtils as ut
from multiprocessing import Pool
from sklearn.utils import shuffle

class ForkingPickler4(multiprocessing.reduction.ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 4
        else:
            args.append(4)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        #print("USING VERSION 4!!!")
        return multiprocessing.reduction.ForkingPickler.dumps(obj, protocol)

class Pickle4Reducer(multiprocessing.reduction.AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    def dump(self, obj, file, protocol=4):
        ForkingPickler4(file, protocol).dump(obj)

ctx = multiprocessing.get_context()
ctx.reducer = Pickle4Reducer()

def augment_data(signals, labels, signal_size,augmentation_factor,ncore):
    if augmentation_factor <= 1:
        print('Not performing data augmentation')
        signals,labels = suffle(signals,labels)
        return signals, labels
    print('n_cores', ncore)
    print('data augmentation: Start ',end='',flush=True)

    results = []
    ntrack = int((augmentation_factor - 1) * len(signals) * 0.02)
    j = 0
    inp_results = []
    for signal,label in zip(signals, labels):
        inp_results.append((signal,label,j,ntrack))
        j += 1
    o_results = [(signal,label) for signal,label in zip(signals, labels)]
    results.extend(o_results)
    for _ in range(augmentation_factor - 1):
        pool = Pool(ncore)
        rs = pool.starmap(modify_signal_l,inp_results,chunksize=1000)
        pool.close()
        pool.join()
        results.extend(rs)
        inp_results = []
        for signal,label in zip(signals, labels):
            inp_results.append((signal,label,j,ntrack))
            j += 1

    print(' end',flush=True)
    sig = []; lab = [];
    for r in results:
        sig.append(r[0])
        lab.append(r[1])
    augmented_signals, augmented_labels =  shuffle(sig,lab)
    #augmented_signals, augmented_labels = margeAndSuffle(results,signal_size)
    print('end merge')
    return augmented_signals, augmented_labels

#@jit(nopython=True)
def margeAndSuffle(results,signal_size):

    datasize = len(results)
    shuffle_indices = np.random.permutation(np.arange(datasize))
    # print("si",shuffle_indices)

    augmented_signals = np.empty([datasize, signal_size], dtype=np.float32)
    augmented_labels  = np.empty(datasize, dtype=int)
    for idx in range(len(shuffle_indices)):
        i = shuffle_indices[idx]
        augmented_signals[idx] = results[i][0]
        augmented_labels[idx]  = results[i][1]

    #augmented_signals,augmented_labels = _merge(x, y, results, shuffle_indices)
    # print("al",augmented_labels)
    return augmented_signals,augmented_labels

def suffle(signals,labels):

    datasize = len(signals)
    labels = np.array(list(labels))
    print(labels)
    shuffle_indices = np.random.permutation(np.arange(datasize))
    sufflex = signals[shuffle_indices]
    suffley = labels[shuffle_indices]
    return sufflex,suffley

#@jit(nopython=True)
@jit
def _merge(x,y,results,shuffle_indices):

    for idx in range(len(shuffle_indices)):
        i = shuffle_indices[idx]
        x[idx] = results[idx][0]
        y[idx] = results[idx][1]
    return x,y


def modify_signal_l(signal, label, idx, ntrack):
    sig = modify_signal(signal)
    unicode_string = '\u2588'
    if idx % ntrack == 0:
        print(unicode_string,end='', flush=True)
    
    return (sig, label)

@jit(nopython=True)
def modify_signal(signal):

    modification_count = len(signal) * 0.3
    modification_count = int(round(modification_count / 2)) * 2 # to int
    #modification_positions = random.sample(range(len(signal)), k=modification_count)
    modification_positions = np.random.choice(len(signal),modification_count)

    half = int(modification_count / 2)
    duplication_positions = set(modification_positions[:half])
    deletion_positions = set(modification_positions[half:])

    new_signal = np.empty(len(signal), np.float32)
    j = 0
    for i, val in enumerate(signal):
        if i in duplication_positions and j < len(signal):

            new_signal[j] = val
            j += 1
            new_signal[j] = val
            j += 1
        elif i in deletion_positions:
            pass
        elif j < len(signal):
            new_signal[j] = val
            j += 1

    return new_signal

