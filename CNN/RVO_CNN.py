import os
import sys
import glob

import argparse
import datetime
from pandas_ml import ConfusionMatrix
from datetime import datetime as T
from configparser import ConfigParser

from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout 

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

################################################

n_batches     = 16
n_epoches     = 30
n_channels    = 3
dropout       = 0.5
learning_rate = 1e-2
random_seed   = 3

################################################

CWD             = os.path.dirname(os.path.abspath(__file__))
FOLDID          = CWD.split("/")[-1]
EXPCODE         = CWD.split("/")[-2]
DIM             = 299
LOGDIR          = os.path.join(CWD, "dir")
LOGNAME         = "log_%s_%s.txt"%(EXPCODE, FOLDID)
LOGFILE         = os.path.join(LOGDIR, LOGNAME)
CSVNAME         = "csv_%s_%s.txt"%(EXPCODE, FOLDID)
CSVFILE         = os.path.join(LOGDIR, CSVNAME)
MODELDIR        = os.path.join(CWD,"dir")
DATADIR         = os.path.join(CWD, "dir")
ROWS, COLS      = DIM, DIM
n_classes       = 2

################################################
def set_callback_list(exp_code):
  ckpt_filepath = 'dir' + exp_code + '_{epoch:04d}.h5'
  checkpoint = ModelCheckpoint(ckpt_filepath,
                               monitor='val_acc',
                               verbose=2,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='max')
                            
  tensorboard = TensorBoard(log_dir='dir', 
                              histogram_freq=0,
                              batch_size=n_batches,
                              write_graph=True,
                              write_images=True)

  csv_filename = 'dir' + exp_code + '.log'
  csv_logger   = CSVLogger(csv_filename, separator='\t', append=False)
  return [checkpoint, tensorboard, csv_logger]

################################################
  
if __name__ == '__main__':
  ST = T.now() 

  train_path       = "dir"
  train_datagen    = ImageDataGenerator(rescale=1./255)
  train_generator  = train_datagen.flow_from_directory(train_path, 
                                        target_size=(ROWS, COLS),
                                        batch_size=n_batches,
                                        class_mode='categorical',
                                        shuffle=True)

  test_path        = "dir"
  test_datagen     = ImageDataGenerator(rescale=1./255)
  test_generator   = test_datagen.flow_from_directory(test_path,
                                       target_size=(ROWS, COLS),
                                       batch_size=n_batches,
                                       class_mode='categorical',
                                       shuffle=False)

  base_model = Xception(weights='imagenet',
                        include_top=False,
                        input_shape=(ROWS, COLS, n_channels)
                        )
                                     
  model_fc = base_model.output
  model_fc = Dropout(dropout)(model_fc)
  model_fc = Dense(2048, activation='relu')(model_fc)
  model_fc = Dropout(dropout)(model_fc)
  model_fc = Dense(2048, activation='relu')(model_fc)
  model_fc = GlobalAveragePooling2D()(model_fc)
  
  model_fc = Dense(units=n_classes, activation='softmax')(model_fc)
  model    = Model(inputs=base_model.input, outputs=model_fc)

  n_layers = len(model.layers)
  for layer in model.layers:
    layer.trainable = True

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  prefix = "%s_%s"%(EXPCODE, FOLDID)
  hist = model.fit_generator(train_generator,
                             steps_per_epoch=len(train_generator)/n_batches,
                             epochs=n_epoches,
                             callbacks=set_callback_list(prefix),
                             validation_data=test_generator,
                             validation_steps=len(test_generator))

  ET = T.now() 
  print("Elapsed Time =", ET-ST)
  






















