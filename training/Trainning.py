import os,sys
import math
from pyarrow import parquet as pq
import numpy as np
from multiprocessing import Pool,Process,Queue
from functools import partial
import random
import glob
import pandas as pd
from tensorflow import keras
import nnmodels.CNNWavenet as cnnwavenet
from numba import jit
import multiprocessing
import csv
from tensorflow.keras.callbacks import ModelCheckpoint
import utils.tyUtils as ut

from training.SignalGenerator import AugGenerator
from training.SignalGenerator import BatchIterator

import tensorflow as tf
import tensorflow_addons as tfa
import datetime,time
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import logging, traceback
import pickle

tf.get_logger().setLevel('ERROR')

logging.disable(logging.WARNING)

FIG_SIZE_WIDTH = 12
FIG_SIZE_HEIGHT = 10
FIG_FONT_SIZE = 25


class CustomCallback(keras.callbacks.Callback):
    
    def _set_batch_size(self,batch_size):
        self.batch_size = batch_size
    
    def _set_requested_epoch(self,epoch,total_epoch,sum_epoch):
        self.num_epoch = epoch
        self.total_epoch = total_epoch
        self.sum_epoch = sum_epoch
    
    def _set_sample_size(self,sample):
        self.num_sample = sample

    def _set_plot(self,plot_batch=True,plot_epoch=True):
        self.plot_batch = plot_batch
        self.plot_epoch = plot_epoch

    def _set_wait(self,t):
        self.t = t

    def on_train_begin(self, logs=None):
        print("Start training")
        self.time_per_batch = None
        self.nbatch = math.ceil(self.num_sample / self.batch_size)
        self.e_loss = []; self.e_acc = []; self.e_val_loss = []; self.e_val_acc = [];

    def on_train_end(self, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        acc = logs['accuracy'] * 100.0
        val_acc = logs['val_accuracy'] * 100.0
        print("Validation statistics after training: Accuracy %.2f Loss %.6f" % (val_acc,val_loss))
        print("Training statistics after training: Accuracy %.2f Loss %.6f" % (acc,loss))

        e = list(range(len(self.e_loss)))

        if self.plot_epoch:
            plt.rcParams["font.size"] = 8 
            plt.subplot(121)
            plt.plot(e,self.e_loss,'k-',label='train')
            plt.plot(e,self.e_val_loss,'b-',label='validation')
            plt.title('Epoch Loss')
            plt.subplot(122)
            plt.plot(e,self.e_acc,'k-',label='train')
            plt.plot(e,self.e_val_acc,'b-',label='validation')
            plt.title('Epoch accuracy')
            plt.show()

        print(flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        print(flush=True)
        time.sleep(self.t)
        self.epoch_time_start = time.time()
        self.b_loss = []
        self.b_acc  = []
        print("Start epoch {}/{} of training, Total Epoch: {}/{}".format(epoch+1,self.num_epoch, \
               epoch+self.total_epoch+1,self.sum_epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_end = math.ceil(time.time() - self.epoch_time_start)
        loss = logs['loss']
        val_loss = logs['val_loss']
        acc = logs['accuracy'] * 100.0
        val_acc = logs['val_accuracy'] * 100.0
        print("\nValidation statistics at Epoch %d: Accuracy %.2f Loss %.6f" % (epoch,val_acc,val_loss))
        print("Training statistics   at Epoch %d: Accuracy %.2f Loss %.6f" % (epoch,acc,loss))

        self.e_loss.append(loss)
        self.e_acc.append(acc)
        self.e_val_loss.append(val_loss)
        self.e_val_acc.append(val_acc)

        self.time_per_batch = int((self.epoch_time_end / self.nbatch) * 1000.0) # ms
        remaining_epoch = self.num_epoch - epoch
        eta_train = self.epoch_time_end * remaining_epoch
        print("Time taken: %d sec, Time/batch: %d ms, Time remaining to train: %d sec" % (self.epoch_time_end,self.time_per_batch,eta_train))
        
        if self.plot_batch:
            b = list(range(len(self.b_loss)))
            plt.rcParams["font.size"] = 8 
            plt.subplot(121)
            plt.plot(b,self.b_loss,'k-')
            plt.title('Batch loss')
            plt.subplot(122)
            plt.plot(b,self.b_acc,'k-')
            plt.title('Batch accuracy')
            plt.show()

        print(flush=True)

    def on_train_batch_end(self, batch, logs=None):
        batch_time = math.ceil(time.time() - self.epoch_time_start) * 1000  # millisecond 
        self.b_loss.append(logs['loss'])
        self.b_acc.append(logs['accuracy'])
        batch_remaining = self.nbatch - batch
        if self.time_per_batch is None:
            if batch > 10:
                tpb = batch_time / batch
                eta = str(math.floor((tpb * batch_remaining) / 1000.0)) # sec
            else: eta = '???'
        else:
            eta = str(math.floor((self.time_per_batch * batch_remaining) / 1000.0)) # sec
        print("\rBatch %6d/%6d, loss %.6f accuracy %.2f ETA: %-10s sec" % (batch+1,self.nbatch, \
                logs['loss'],logs['accuracy']*100,eta),end="")

    def on_test_begin(self, logs=None):
        self.test_time_start = time.time()

    def on_test_end(self,logs=None):
        self.test_time = math.ceil(time.time() - self.test_time_start)
        

class Trainer:

    def __init__(self,indir,outdir,paramPath,limit=None,unit=1000,split=0.2,          \
                 epoch=50,total_epoch=0,sum_epoch=50,data_augment=0,dropoutRate=0.2,  \
                 test_time_per_sample=0,cross_fold=False,k=None,iteration=0):
        self.dirpath = indir
        self.outdir = outdir
        self.paramPath = paramPath
        self._set_param(paramPath)
        self._set_epoch(epoch,total_epoch,sum_epoch)
        self._set_augmentation(data_augment)
        self._set_dropout(dropoutRate)
        self._set_dataLimit(limit)
        self._set_dataUnit(unit)
        self._set_dataSplit(split)
        self._set_iter(iteration)
        self._set_test_time_per_sample(test_time_per_sample)
        self._set_cross(cross_fold,k)

        self._set_output_files()

    def set_memory_growth(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for gpu_instance in physical_devices:
            tf.config.experimental.set_memory_growth(gpu_instance, True)

    def _set_param(self,paramPath):
        self.param = ut.get_parameter(paramPath)

    def _set_epoch(self,epoch,total_epoch,sum_epoch):
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.sum_epoch = sum_epoch

    def _set_augmentation(self,level):
        self.augmentationLevel = level

    def _set_dropout(self,dropoutRate):
        self.dropoutRate = dropoutRate

    def _set_dataLimit(self,dataLimit,defaultDataLimit=12000):
        if dataLimit is None:
            self.dataLimit = defaultDataLimit
        else:
            self.dataLimit = dataLimit
        self.defaultDataLimit = defaultDataLimit

    def _set_dataUnit(self,unit):
        self.dataUnit = unit

    def _set_dataSplit(self,split):
        self.dataSplit = split

    def set_batchSize(self,batchSize = 128):
        self.batchSize = batchSize

    def _set_iter(self,iteration):
        self.iter = iteration

    def _set_test_time_per_sample(self,test_time_per_sample):
        self.test_time_per_sample = test_time_per_sample

    def _set_cross(self,cross_fold,k):
        self.cross_fold = cross_fold
        self.k = k

    def set_loss(self,loss='categorical_crossentropy',metric=None):
        self.loss = loss
        self.metrics = metric

    def formatX(self,X,wlen,filter_index=None):
        Xf = np.reshape(X, (-1, wlen, 1))
        if filter_index is None:
            return Xf
        Xf = Xf[np.array(filter_index),:,:]
        return Xf

    def formatY(self,Y,num_classes,filter_index=None):
        Yf = np.reshape(Y, (-1, 1,))
        if filter_index is None:
            return keras.utils.to_categorical(Yf, num_classes)
        Yf = Yf[np.array(filter_index),:]
        return keras.utils.to_categorical(Yf, num_classes)

    def build(self):

        self.model = cnnwavenet.build_network(shape=(None, self.wlen, 1), num_classes=self.num_classes, do_r = self.dropoutRate)

    def show_model(self):
        self.model.summary()

    @staticmethod
    def get_lr(i,m,c):
        return c - i*m

    def get_learning_rates(self,epoch):
        c1 = self.base_lr; m1 = c1 * 0.1;
        l1 = [Trainer.get_lr(i,m=m1,c=c1) for i in range(10)]
        c2 = min(l1); m2 = c2 * 0.1;
        l2 = [Trainer.get_lr(i,m=m2,c=c2) for i in range(10)]
        ll = l1 + l2
        min_lr = min(ll)
        
        lr = [None] * epoch
        for i in range(epoch):
            if i < len(ll):
                lr[i] = ll[i]
            else:
                lr[i] = min_lr
        return lr

    def decay_schedule(self,epoch):

        return self.base_lr # no decay

        epoch = epoch + self.total_epoch
        if epoch > 1 and epoch <= 5:
            return self.base_lr * 0.25                   # 0.0002
        elif epoch > 5 and epoch <= 20:
            return (self.base_lr * 0.25)/2.0             # 0.0001
        elif epoch > 20 and epoch <= 50:
            return ((self.base_lr * 0.25)/2.0)/2.0       # 0.00005
        elif epoch > 50:
            return (((self.base_lr * 0.25)/2.0)/2.0)/5.0 # 0.00001
        else:
            return self.base_lr                          # 0.0008
        #return self.learning_rate_precomputed[epoch]
        
    def set_optimizer(self,learning_rate=0.0008):

        self.base_lr = learning_rate
        optim = keras.optimizers.Adam(learning_rate=self.base_lr, 
                                      beta_1=0.9, beta_2=0.999, 
                                      epsilon=None, decay=0.0, amsgrad=False, 
                                      clipnorm=1.0)
        if self.augmentationLevel > 0:
            optim = tfa.optimizers.SWA(optim)

        self.optim = optim

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optim, metrics=self.metrics)

    def makeFileName(self,filename,extension):
        filename_save = filename
        if self.iter >= 1:
            filename = filename + "_v" + str(self.iter)
            if self.cross_fold:
                filename = filename + "_k" + str(self.k)
            if self.iter == 1:
                p_filename = filename_save
                if self.cross_fold:
                    p_filename = p_filename + "_k" + str(self.k)
            else:
                p_iter = self.iter - 1
                p_filename = filename_save + "_v" + str(p_iter)
                if self.cross_fold:
                    p_filename = p_filename + "_k" + str(self.k)
        else:
            p_filename = None
            if self.cross_fold:
                filename = filename + "_k" + str(self.k)
        filename_full   = self.outdir + "/" +   filename + "." + extension
        if p_filename is not None:
            p_filename_full = self.outdir + "/" + p_filename + "." + extension
        else: p_filename_full = None
        print("Output set: ",filename_full)
        print("Previus output: ",p_filename_full)
        return filename_full, p_filename_full

    def _set_output_files(self):
        self.previous_filename = {}
        print("setting output at iter: ",self.iter)
        self.outweight_0,self.previous_filename[0] = self.makeFileName("BestWt","h5")
        self.outweight_1,self.previous_filename[1] = self.makeFileName("AllWt","h5")
        self.historypath,self.previous_filename[2] = self.makeFileName("history","csv")
        self.graphpath,  self.previous_filename[3] = self.makeFileName("history","png")
        self.datapath    = self.outdir + "/data.pkl"

    def _get_output_files(self,previous=False):
        if previous:
            return self.previous_filename[0],self.previous_filename[1],self.previous_filename[2],  \
                    self.previous_filename[3],self.datapath
        return self.outweight_0, self.outweight_1, self.historypath, self.graphpath,self.datapath

    def load_pretrain(self,outPath):
        self.model.load_weights(outPath)

    def set_checkpoints(self,plot_batch=True,plot_epoch=True):

        lr_scheduler = LearningRateScheduler(self.decay_schedule)

        #tqdm_pb = tfa.callbacks.TQDMProgressBar()

        mc_best = ModelCheckpoint(filepath=self.outweight_0, monitor='val_accuracy',
                              verbose=0,save_best_only=True,
                              save_weights_only=True,
                              mode='max')
        mc_every_epoch = ModelCheckpoint(filepath=self.outweight_1, monitor='val_accuracy',
                                      verbose=0,save_weights_only=True,mode='max',
                                      save_frequency=1)

        log_dir = self.outdir + "/log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        ccb = CustomCallback()
        ccb._set_batch_size(self.batchSize)
        ccb._set_requested_epoch(self.epoch,self.total_epoch,self.sum_epoch)
        ccb._set_sample_size(self.num_training_sample)
        ccb._set_wait(0)
        ccb._set_plot(plot_batch=plot_batch,plot_epoch=plot_epoch)
        
        self.cb = [lr_scheduler,mc_best,mc_every_epoch,tensorboard_cb,ccb]

    def read_pqt(self,f,p,q,limit):
        pqt = pq.read_table(f,columns=['read_id', 'trna','trimsignal'])
        dfp = pqt.to_pandas()
        x_train = [];y_train = []
        x_test  = [];y_test  = []
        wlen = 0
        cnt = 0
        for idx, row in dfp.iterrows():
            trna = row[1]
            signal = row[2]
            if wlen == 0:
                wlen = len(signal)
            if cnt % p >= q:
                x_train.append(signal)
                y_train.append(trna)
            else:
                x_test.append(signal)
                y_test.append(trna)

            if limit is not None:
                if cnt >= limit - 1:
                    break
            cnt += 1
        
        trna = list(dfp["trna"].unique())

        return (x_train,y_train,x_test,y_test,trna,wlen)

    def get_train_test_index(self,n_tot,n_test):

        random_valid_index = random.sample(range(0,n_tot),n_test)
        random_train_index = [i for i in range(n_tot) if i not in random_valid_index]
        return random_train_index, random_valid_index

    def read_pqt_cross(self,filename,split,limit):
        
        pqt = pq.read_table(filename,columns=['read_id', 'trna','trimsignal'])
        dfp = pqt.to_pandas()
        random_index = random.sample(range(0,len(dfp)),limit)
        dfp_lim = dfp.iloc[random_index,:]

        K = int(1.0 / split)
        dfp_set = np.array_split(dfp_lim,K)
        x_set = [None] * K; y_set = [None] * K; lab_set = [None] * K;
        for k in range(K):
            x_set[k] = list(dfp_set[k]['trimsignal'])
            y_set[k] = list(dfp_set[k]['trna'])
            lab_set[k] = list(dfp_set[k]['read_id'])
        trna = list(np.unique(dfp_lim['trna']))
        wlen = len(x_set[0][0])

        tup = (x_set,y_set,lab_set,trna,wlen)

        return tup

    def read_pqt_simple(self,filename,split,limit):

        pqt = pq.read_table(filename,columns=['read_id', 'trna','trimsignal'])
        dfp = pqt.to_pandas()
        trnas = []; wlen = 0;
        random_index = random.sample(range(0,len(dfp)),limit)
        dfp_lim = dfp.iloc[random_index,:]
        n_valid = int(len(dfp_lim) * split)
        
        tid, vid = self.get_train_test_index(len(dfp_lim),n_valid)
        dfp_valid = dfp_lim.iloc[vid,:].copy()
        dfp_train = dfp_lim.iloc[tid,:].copy()
        x_valid = list(dfp_valid['trimsignal'])
        y_valid = list(dfp_valid['trna'])
        label_valid = list(dfp_valid['read_id'])
        x_train = list(dfp_train['trimsignal'])
        y_train = list(dfp_train['trna'])
        label_train = list(dfp_train['read_id'])
            
        trna = list(np.unique(y_train + y_valid))
            
        wlen = len(x_valid[0])

        tup = (x_train,y_train,label_train,x_valid,y_valid,label_valid,trna,wlen)

        return tup

    def load_cross(self):
        fs = glob.glob(self.dirpath + "/*.pq*")
        
        wlen = 0;
        K = int(1.0 / self.dataSplit)
        X_set = [None] * K; Y_set = [None] * K; Lab_set = [None] * K;
        for k in range(K):
            X_set[k] = []; Y_set[k] = []; Lab_set[k] = [];
        trnas = []
        ncore = self.param.ncore
        ro = []
        with Pool(ncore) as pool:
            for f in fs:
                print(f)
                ro.append(pool.apply_async(self.read_pqt_cross,args=(f,self.dataSplit,self.dataLimit)))
            pool.close()
            pool.join()
        results = [p.get() for p in ro]
        self.num_training_sample = 0
        for result in results:
            x_set, y_set, lab_set, ts, wlen = result
            for k in range(K):
                X_set[k].extend(x_set[k]); Y_set[k].extend(y_set[k]); 
                Lab_set[k].extend(lab_set[k]);
                self.num_training_sample += len(x_set[k])
            trnas.extend(ts)
        trnas = sorted(trnas)

        trnas = dict([(v,k) for k,v in enumerate(sorted(np.unique(trnas)))])
        
        # name to index
        for k in range(K):
            Y_set[k] = [trnas[y] for y in Y_set[k]]

        num_classes = len(trnas.keys())
        print("Number of classes: ",num_classes)

        self.X_train = X_set
        self.X_test = None
        self.Y_train = Y_set
        self.Y_test = None
        self.trnas = trnas
        self.num_classes = num_classes
        self.wlen = wlen

        self.num_validatn_sample = None
        print("Number of sample: ", self.num_training_sample)

    def load_simple(self):
        fs = glob.glob(self.dirpath + "/*.pq*")
        trnas = []
        X_train = []; Y_train = []
        X_test = []; Y_test = []
        wlen = 0

        ncore = self.param.ncore
        ro = []
        with Pool(ncore) as pool:
            for f in fs:
                print(f)
                ro.append(pool.apply_async(self.read_pqt_simple,args=(f,self.dataSplit,self.dataLimit)))
            pool.close()
            pool.join()
        results = [p.get() for p in ro]
        for result in results:
            x_train,y_train,label_train,x_valid,y_valid,label_train,ts,wlen = result
            X_train.extend(x_train);Y_train.extend(y_train)
            X_test.extend(x_valid);Y_test.extend(y_valid)
            trnas.extend(ts)
        trnas = sorted(trnas)

        trnas = dict([(v,k) for k,v in enumerate(sorted(np.unique(trnas)))])

        # name to index
        Y_train = [trnas[y] for y in Y_train]
        Y_test  = [trnas[y] for y in Y_test ]

        #Y_train = list(map(lambda trna: trnas.index(trna), Y_train))
        #Y_test = list(map(lambda trna: trnas.index(trna), Y_test))
        #num_classes = np.unique(Y_train).size

        num_classes = len(trnas.keys())
        print("Number of classes: ",num_classes)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.trnas = trnas
        self.num_classes = num_classes
        self.wlen = wlen

        self.num_training_sample = len(self.X_train)
        self.num_validatn_sample = len(self.X_test)

    def load(self):

        fs = glob.glob(self.dirpath + "/*.pq*")
        trnas = []
        X_train = []; Y_train = []
        X_test = []; Y_test = []
        wlen = 0
        
        p = int(self.dataLimit / self.dataUnit)
        q = int(p * self.dataSplit)

        ncore = self.param.ncore
        ro = []
        with Pool(ncore) as pool:
            for f in fs:
                print(f)
                ro.append(pool.apply_async(self.read_pqt,args=(f,p,q,self.dataLimit)))
            pool.close()
            pool.join()
        results = [p.get() for p in ro]
        for result in results:
            x_train,y_train,x_test,y_test,ts,wlen = result
            X_train.extend(x_train)
            Y_train.extend(y_train)
            X_test.extend(x_test)
            Y_test.extend(y_test)
            trnas.extend(ts)
        #trnas = list(np.unique(trnas))
        trnas = dict([(v,k) for k,v in enumerate(sorted(np.unique(trnas)))])
        
        # name to index
        Y_train = [trnas[y] for y in Y_train] 
        Y_test  = [trnas[y] for y in Y_test ] 

        #Y_train = list(map(lambda trna: trnas.index(trna), Y_train))
        #Y_test = list(map(lambda trna: trnas.index(trna), Y_test))
        #num_classes = np.unique(Y_train).size

        num_classes = len(trnas.keys())
        print("Number of classes: ",num_classes)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.trnas = trnas
        self.num_classes = num_classes
        self.wlen = wlen

        self.num_training_sample = len(self.X_train)
        self.num_validatn_sample = len(self.X_test)

    def savedata(self,datapath):
        with open(datapath,'wb') as file:
            pickle.dump((self.X_train,self.X_test,self.Y_train,self.Y_test,self.trnas,self.num_classes,
                         self.wlen,self.num_training_sample,self.num_validatn_sample),file,pickle.HIGHEST_PROTOCOL)

    def loadSavedData(self,filename):
        with open(filename,'rb') as file:
            self.X_train,self.X_test,self.Y_train,self.Y_test,self.trnas,self.num_classes, \
                    self.wlen,self.num_training_sample,self.num_validatn_sample = pickle.load(file)

    def prepare_train_test(self,apply_post_filter_to_train=False):

        if self.cross_fold:
            test_x = self.formatX(self.X_train[self.k], self.wlen)
            test_y = self.formatY(self.Y_train[self.k], self.num_classes)
            train_x_set = []; train_y_set = [];
            for ktr in range(len(self.X_train)):
                if ktr != self.k:
                    train_x_set.append(self.formatX(self.X_train[ktr], self.wlen))
                    train_y_set.append(self.formatY(self.Y_train[ktr], self.num_classes))
            train_x = np.concatenate(train_x_set,axis=0)
            train_y = np.concatenate(train_y_set,axis=0)
            if apply_post_filter_to_train:
                train_x = train_x[np.array(filter_index),:,:]
                train_y = train_y[np.array(filter_index),:]
        else:
            test_x = self.formatX(self.X_test, self.wlen)
            test_y = self.formatY(self.Y_test, self.num_classes)
            if apply_post_filter_to_train:
                train_x, train_y = self.apply_post_filter_to_train(filter_index)
            else:
                train_x = self.formatX(self.X_train,self.wlen)
                train_y = self.formatY(self.Y_train,self.num_classes)
        print('test_x shape:', test_x.shape)
        print('test_y shape:', test_y.shape)
        print('train_x shape:', train_x.shape)
        print('train_y shape:', train_y.shape)

        return train_x,train_y,test_x,test_y

    def fit(self,filter_index = None, apply_post_filter_to_train=False):

        train_x,train_y,test_x,test_y = self.prepare_train_test(apply_post_filter_to_train=apply_post_filter_to_train)

        if self.augmentationLevel == 0:

            self.num_training_sample = train_x.shape[0]
            self.num_validatn_sample = test_x.shape[0]
            self.cb[4]._set_sample_size(self.num_training_sample)

            history = self.model.fit(train_x, train_y, epochs=self.epoch, batch_size=self.batchSize,verbose=0,
                                     shuffle=True, validation_data=(test_x, test_y),callbacks=self.cb)
        else:

            signalgen = AugGenerator(train_x, train_y, self.batchSize,self.wlen,
                                           self.num_classes,self.augmentationLevel,
                                           self.epoch,self.param.ncore)
            batchgen = BatchIterator(test_x, test_y, self.batchSize,self.wlen,
                                     self.num_classes,self.epoch)
            n_aug_batch  = signalgen.numbatch()
            n_aug_sample = signalgen.numsample()
            print("fitting with augmentation with %d batches" % n_aug_batch) 

            t  = signalgen.get_augment_time()
            print(type(self.test_time_per_sample),self.test_time_per_sample)
            print(type(test_x.shape[0]),test_x.shape[0])
            tt = (self.test_time_per_sample * test_x.shape[0]) / 1000
            print("Augmentation Time  : ",t)
            print("Predicted test time: ",tt)
            delay1 = 2*t + tt
            delay2 = 4*tt
            self.cb[4]._set_sample_size(n_aug_sample)
            self.cb[4]._set_wait(delay1)
            signalgen._set_wait(delay2)

            history = self.model.fit(signalgen.flow(),steps_per_epoch=signalgen.numbatch(),
                                     validation_data=batchgen.flow(),
                                     validation_steps=batchgen.numbatch(),verbose=0,
                                     epochs=self.epoch,callbacks=self.cb)
        self.history = history

    def apply_post_filter_to_train(self,filter_index):
        x = self.formatX(self.X_train, self.wlen,filter_index=filter_index)
        y = self.formatY(self.Y_train, self.num_classes,filter_index=filter_index)
        return x,y

    def do_predict_on_batches(self,x,batch_size=None,verbose=0,steps=None):

        # custom batched prediction loop to avoid memory leak issues for now in the model.predict call
        # https://github.com/keras-team/keras/issues/13118
        n = len(x)
        if batch_size is None: batch_size = self.batchSize
        y_pred_probs = np.empty([n, self.num_classes], dtype=np.float32)  # pre-allocate required memory for array for efficiency

        bindex = np.arange(start=0, stop=n, step=batch_size)  # row indices of batches
        bindex = np.append(bindex, n)  # add final batch_end row
        nBatch = len(bindex)
        for index in np.arange(nBatch - 1):
            batch_start = bindex[index]  # first row of the batch
            batch_end   = bindex[index + 1]  # last row of the batch
            y_pred_probs[batch_start:batch_end,:] = self.model.predict_on_batch(x[batch_start:batch_end,:,:])
        return y_pred_probs

    def evaluate(self, datatype='test', dataset = None, threshold=0.0):

        train_x,train_y,test_x,test_y = self.prepare_train_test()

        if datatype == 'test' or datatype == 'train':
            train_x,train_y,test_x,test_y = self.prepare_train_test()
            if datatype == 'test':
                x = test_x
                y = test_y
            else:
                x = train_x
                y = train_y
        else:
            x = dataset[0]
            y = dataset[1]
        N = len(x)
        prediction = self.do_predict_on_batches(x, batch_size=self.batchSize, verbose=0, steps=None)
        #prediction = self.model.predict(x, batch_size=self.batchSize, verbose=0, steps=None)

        index = 0; acc = 0;
        eval_index = []
        for row in prediction:
            rdata = np.array(row)
            max_prob = rdata.max()
            if max_prob < threshold:
                continue
            p_id = np.argmax(rdata)
            o_id = np.argmax(y[index,:])
            if p_id == o_id: acc += 1
            eval_index.append(index)
            index += 1
        percent_accuracy   = 100.0 * (acc / index)
        percent_throughput = 100.0 * (index / N)
        eval_index = np.array(eval_index)
        print("Throughput: ",percent_throughput)
        print("Accuracy  : ",percent_accuracy)
        return eval_index,percent_accuracy,percent_throughput

    def write_class(self):
        trnapath = self.outdir + '/tRNAindex.csv'
        with open(trnapath, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerows(self.trnas)

    def write_history(self):

        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(self.historypath)

    def plot_history(self):

        ut.plot_history(self.history,save_graph_img_path=self.graphpath,
                        fig_size_width=FIG_SIZE_WIDTH,
                        fig_size_height=FIG_SIZE_HEIGHT,
                        lim_font_size=FIG_FONT_SIZE)

    def get_test_time_mean(self):
        test_time_ms = self.cb[4].test_time * 1000
        test_size = self.num_validatn_sample
        return test_time_ms / test_size

def do_train(indir,outdir,paramPath,iteration,epoch = 50,total_epoch = 0,sum_epoch=50,
             data_augment = 0, dropoutRate = 0.2,learning_rate = 0.00080,
             batch_size=128,limit = None, unit = 1000, split=0.2, 
             test_time_per_sample = 0, post_threshold = 0.0, 
             done = False, evaluateOnly = False, 
             filter_index = None, apply_post_filter_to_train = False,
             cross_fold = False, k = None, load_with_shuffle = True,
             datapath = None, wlen = None, num_classes = None):

    print("Doing scheme: ", iteration)
    T = Trainer(indir, outdir, paramPath, limit=limit, unit=unit, split=split,
                epoch=epoch, total_epoch=total_epoch, sum_epoch=sum_epoch,
                data_augment=data_augment, dropoutRate=dropoutRate, cross_fold = cross_fold, k = k,
                test_time_per_sample=test_time_per_sample, iteration=iteration)

    outfiles_prev = T._get_output_files(previous=True)
    outfiles_curr = T._get_output_files(previous=False)

    if done:
        datapath = outfiles_curr[4]
        print("Loading data from ",datapath)
        T.loadSavedData(datapath)
        wlen = T.wlen
        num_classes = T.num_classes

        T.set_memory_growth()

        print("Building network")
        T.build()

        print("Setting optim and batch size")
        T.set_optimizer(learning_rate = learning_rate); T.set_batchSize(batch_size)

        print("Setting loss function")
        T.set_loss(loss='categorical_crossentropy',metric=['accuracy'])

        print("Compile model and load pretrain (opt)")
        T.compile()
        T.load_pretrain(outfiles_curr[0])
        
        print("Evaluate")
        eids,acc,thr =  T.evaluate(threshold=post_threshold)
        return datapath, eids, None, wlen, num_classes

    if iteration > 0:
        print("Loading data from ",datapath)
        T.loadSavedData(datapath)
    else:
        # avoid_loading_by_check
        if datapath is not None and os.path.isfile(datapath):
            print("Found data for loading at ",datapath)
            T.loadSavedData(datapath)
        else:
            if load_with_shuffle:
                print("Loading data with shuffle")
                T.load_simple()
            else:
                if cross_fold:
                    print("Loading data for ensemble learning")
                    T.load_cross()
                else:
                    print("Loading data with original method")
                    T.load()
            print("Loaded input data is saved in %s" % outfiles_curr[4])
            datapath = outfiles_curr[4]
            T.savedata(datapath)

    print("Building network")
    T.build()
    print("Setting optim and batch size")
    T.set_optimizer(learning_rate = learning_rate); T.set_batchSize(batch_size)
    print("Setting loss function")
    T.set_loss(loss='categorical_crossentropy',metric=['accuracy'])
    print("Compile model and load pretrain (opt)")
    T.compile()

    if iteration > 0: 
        print("Loaded pretrained model weights from %s" % outfiles_prev[0])
        T.load_pretrain(outfiles_prev[0])
    
    print("Setting checkpointers")
    T.set_checkpoints(plot_batch=True,plot_epoch=True)

    if not evaluateOnly:
        if apply_post_filter_to_train:
            print("Preparing for post filtering before next train")
            eids,acc,thr =  T.evaluate(datatype='train',threshold=post_threshold)
        else:
            eids = None
        print("Start fitting")
        T.fit(filter_index = eids, apply_post_filter_to_train=apply_post_filter_to_train)

        print("Outputting")
        T.write_class()
        T.write_history()
        T.plot_history()
        test_time_per_sample = T.get_test_time_mean()
        wlen = T.wlen
        num_classes = T.num_classes

    print("Evaluate")
    eids,acc,thr =  T.evaluate(threshold=post_threshold)

    return datapath, eids, test_time_per_sample,wlen,num_classes

def process_run(func, *args,**kwargs):
    def wrapper_func(queue, *args,**kwargs):
        queue.put(func(*args,**kwargs))

    def process(*args,**kwargs):
        queue = Queue()
        p = Process(target = wrapper_func, args = [queue] + list(args), kwargs = kwargs)
        p.start()
        result = queue.get()
        p.join()
        return result

    return process(*args,**kwargs)

def do_train_by_scheme(indir,outdir,paramPath,scheme):
    sk = list(sorted(scheme.keys()))
    nscheme = len(sk)
    iteration = 0; 
    sum_epoch = 0
    for i in range(nscheme):
        sum_epoch += scheme[sk[i]]['epoch']
    for i in range(nscheme): scheme[sk[i]]['sum_epoch'] = sum_epoch
    scheme[sk[iteration]]['total_epoch'] = 0
    scheme[sk[iteration]]['test_time_per_sample'] = 0

    if scheme[sk[iteration]]['cross_fold']:
        n_cross = int(1.0 / scheme[sk[iteration]]['split'])
        for k in range(n_cross):
            scheme[sk[iteration]]['k'] = k
            print("Training for fold: ",k)
            datapath, eids, test_time_per_sample,wlen, num_classes = process_run(do_train,indir,outdir,paramPath,iteration,**scheme[sk[iteration]])
            scheme[sk[iteration]]['datapath'] = datapath
            scheme[sk[iteration]]['wlen'] = wlen
            scheme[sk[iteration]]['num_classes'] = num_classes
            scheme[sk[iteration]]['filter_index'] = eids
    else:
        datapath, eids, test_time_per_sample,wlen, num_classes = process_run(do_train,indir,outdir,paramPath,iteration,**scheme[sk[iteration]])
        scheme[sk[iteration]]['datapath'] = datapath
        scheme[sk[iteration]]['wlen'] = wlen
        scheme[sk[iteration]]['num_classes'] = num_classes
        scheme[sk[iteration]]['filter_index'] = eids
    print("Datapath: %s" % datapath)
    for iteration in range(1,nscheme):
        scheme[sk[iteration]]['datapath'] = datapath
        scheme[sk[iteration]]['wlen'] = wlen
        scheme[sk[iteration]]['num_classes'] = num_classes
        scheme[sk[iteration]]['filter_index'] = eids
        te_prev = 0
        for it2 in range(iteration):
            te_prev += scheme[sk[it2]]['epoch']
        scheme[sk[iteration]]['total_epoch'] = te_prev 
        scheme[sk[iteration]]['test_time_per_sample'] = test_time_per_sample 
        if scheme[sk[iteration]]['cross_fold']:
            n_cross = int(1.0 / scheme[sk[iteration]]['split'])
            for k in range(n_cross):
                scheme[sk[iteration]]['k'] = k
                print("Training for fold: ",k)
                datapath, eids, test_time_per_sample,wlen, num_classes = process_run(do_train,indir,outdir,paramPath,iteration,**scheme[sk[iteration]])
        else:
            datapath,eids,test_time_per_sample,wlen, num_classes = process_run(do_train,indir,outdir,paramPath,iteration,**scheme[sk[iteration]])
        print(iteration,scheme[sk[iteration]]['total_epoch'], scheme[sk[iteration]]['sum_epoch'])
