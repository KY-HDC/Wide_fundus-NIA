import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import random
import h5py
import json
import argparse
import datetime
from pandas_ml import ConfusionMatrix
from datetime import datetime as T
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

import matplotlib
import matplotlib.pylab as plt
from configparser import ConfigParser

from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_v3 import InceptionV3

from keras.models import Model
from keras.models import load_model
from keras.models import Input
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.utils import np_utils
from keras.utils.training_utils import multi_gpu_model

################################################

n_batches     = 64
n_epoches     = 100
n_channels    = 3
dropout       = 0.5
learning_rate = 1e-4
random_seed   = 3

################################################

CWD             = os.path.dirname(os.path.abspath(__file__))
NOC             = 2 
FOLDID          = 1 
EXPCODE         = 1 
DIM             = 299
LOGDIR          = os.path.join(CWD, "dir")
LOGNAME         = "log_%s_%s.txt"%(EXPCODE, FOLDID)
LOGFILE         = os.path.join(LOGDIR, LOGNAME)
CSVNAME         = "csv_%s_%s.txt"%(EXPCODE, FOLDID)
CSVFILE         = os.path.join(LOGDIR, CSVNAME)
MODELDIR        = os.path.join(CWD,"dir")
DATADIR         = os.path.join(CWD, "dir")
ROWS, COLS      = DIM, DIM
n_classes       = NOC

################################################

def get_yt(stage):

  if NOC == 2:
    if stage == 'NN': yt = '0'
    else            : yt = '1'
  else:
    if   stage == 'NN': yt = '0'
    elif stage == 'G0': yt = '1'
    elif stage == 'G1': yt = '1'
    elif stage == 'G2': yt = '2'
    elif stage == 'G3': yt = '2'

  return yt

################################################
def get_probs(output, idx):

  if NOC == 2:
    p0 = "%0.5f"%(output[idx, 0])
    p1 = "%0.5f"%(output[idx, 1])
    probs = [p0, p1]
  else:
    p0 = "%0.5f"%(output[idx, 0])
    p1 = "%0.5f"%(output[idx, 1])
    p2 = "%0.5f"%(output[idx, 2])
    probs = [p0, p1, p2]

  return probs
################################################
def run_main(model_name):
 
  exp  = 'test' 
  fold = 'test' 

  model = load_model(model_name)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  test_path        = "dir"
  test_datagen     = ImageDataGenerator(rescale=1./255)
  test_generator   = test_datagen.flow_from_directory(test_path,
                                       target_size=(ROWS, COLS),
                                       batch_size=n_batches,
                                       class_mode='categorical',
                                       shuffle=False)

  print('--Evaluate--')
  scores = model.evaluate_generator(test_generator,steps=len(test_generator))
  val_loss = '%0.5f' %(scores[0])
  val_acc  = '%0.5f' %(scores[1]*100)
  print('val_%s: %0.5f' %(model.metrics_names[0], scores[0]))
  print('val_%s: %0.5f' %(model.metrics_names[1], scores[1]*100)) 
  
  print('--Predict--')
  output = model.predict_generator(test_generator,steps=len(test_generator))
  np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

  jpgs = test_generator.filenames

  table = []
  for i in range(len(jpgs)):
    fname  = os.path.split(jpgs[i])[1]
    prefix = fname.replace('.JPG', '')
    tags   = prefix.split('_')
    yt     = 100 
    lr    = tags[-5] 
    ft    = tags[-4]
    sc    = tags[-3] 
    rt    = tags[-2] 
    color = tags[-1] 
    yp = str(output[i].argmax())
    ox = 0
    if yt == yp: ox = 1
    probs = get_probs(output, i)
    record = [exp, fold, fname, yt, yp, ox, lr, ft, sc, rt, color] + probs
    
    table.append(record)

  DFC_Base = ['EXP', 'FOLD', 'JPG', 'YT','YP','OX', 'LR', 'FILTER', 'SCALE', 'ROTATION', 'COLOR']
  
  if NOC == 2:
    DFC = DFC_Base + ['P0', 'P1']
  else:
    DFC = DFC_Base + ['P0', 'P1', 'P2']

  df = pd.DataFrame(table, columns=DFC)
  df.sort_values(by='JPG', inplace=True)
  predict_file = exp+'_'+fold+'.txt'
  df.to_csv(predict_file, sep='\t', index=None)

#########################################
if __name__ == '__main__':
  ST = T.now()

  best_model = 'dir'
  run_main(best_model)

  ET = T.now()
  print("Elapsed Time =", ET-ST)
