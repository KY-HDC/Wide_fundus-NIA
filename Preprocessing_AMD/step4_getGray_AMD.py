import os
from datetime import datetime as T
import numpy as np
import cv2
import glob
import pandas as pd
import shutil
from joblib import Parallel, delayed

base_path  = 'dir'
step4_path = os.path.join(base_path, 'dir')
step5_path = os.path.join(base_path, 'dir')

######################
def make_subdir(path):
  if os.path.exists(path):
    try:
      shutil.rmtree(path)
    except OSError as ex:
      print(ex)
      exit(0)

  try:
    os.mkdir(path)
  except OSError:
    exit(0)
######################
def get_red_free_image(pImg):
  redFreeImage  = pImg[:, :, 1]
  return redFreeImage
######################
def save_image(pImg, path, srcFName):
  fileName = os.path.join(path, srcFName)
  cv2.imwrite(fileName, pImg)
  print('Saved:', fileName)
######################
def do_get_red_free_image(pFileName):
  _, fileName = os.path.split(pFileName)
  prefix, ext = fileName.split('.')

  imgOrg = cv2.imread(pFileName)

  newFileName = prefix + '_C.' + ext
  imgNew = imgOrg
  save_image(imgNew, step5_path, newFileName)

  newFileName = prefix + '_G.' + ext
  imgNew = get_red_free_image(imgOrg)
  save_image(imgNew, step5_path, newFileName)
######################
def run_main():
  query    = os.path.join(step4_path, '*.JPG')
  fileList = glob.glob(query)
  fileList.sort()

  make_subdir(step5_path)

  Parallel(n_jobs=-1)(delayed(do_get_red_free_image)
                      (fileName) for fileName in fileList)

######################
if __name__ == '__main__':
  ST = T.now()
  run_main()
  ET = T.now()
  print('Elapsed Time =', ET-ST)
