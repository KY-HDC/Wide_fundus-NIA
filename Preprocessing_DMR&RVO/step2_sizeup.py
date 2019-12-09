import os
from datetime import datetime as T
import numpy as np
import cv2
import glob
import pandas as pd
import shutil
from joblib import Parallel, delayed

S140 = 140
S256 = 256
S299 = 299
S512 = 512
DIM  = S299

base_path  = 'dir'
step2_path = os.path.join(base_path, 'dir')
step3_path = os.path.join(base_path, 'dir')

JOBCODE   = ['O', 'U']

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
def resize_image(pImg, SIZE):
  imgRS = cv2.resize(pImg, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
  return imgRS
######################
def save_image(pImg, path, srcFName):
  fileName = os.path.join(path, srcFName)
  cv2.imwrite(fileName, pImg)
  print('Saved:', fileName)
######################
def get_square_image(pImg):
  (h, w) = pImg.shape[:2]
  r = int(w * 0.09)
  y0 = r
  y1 = h - r
  x0 = r
  x1 = w - r

  pImgSQ = pImg[y0:y1, x0:x1, :]

  return pImgSQ
######################
def do_job_BR(pImg, SIZE):
  pImgNew = resize_image(pImg, SIZE)
  return pImgNew
######################
def do_job_BU(pImg, SIZE):
  pImgSQ = get_square_image(pImg)
  pImgNew = resize_image(pImgSQ, SIZE)
  return pImgNew
######################
def do_job(pImg, pJobCode, SIZE):
  if   pJobCode == JOBCODE[0]:
    pImgNew = do_job_BR(pImg, SIZE)
  elif pJobCode == JOBCODE[1]:
    pImgNew = do_job_BU(pImg, SIZE)
  return pImgNew
######################
def do_rescale(pFileName):
  _, fileName = os.path.split(pFileName)
  prefix, ext = fileName.split('.')

  imgOrg = cv2.imread(pFileName)

  for jobCode in JOBCODE:
    newFileName = prefix + '_' + jobCode + '.' + ext
    imgNew = do_job(imgOrg, jobCode, DIM)
    save_image(imgNew, step3_path, newFileName)
######################
def run_main():
  query    = os.path.join(step2_path, '*.JPG')
  fileList = glob.glob(query)
  fileList.sort()

  make_subdir(step3_path)

  Parallel(n_jobs=-1)(delayed(do_rescale)
                      (fileName) for fileName in fileList)

######################
if __name__ == '__main__':
  ST = T.now()
  run_main()
  ET = T.now()
  print('Elapsed Time =', ET-ST)
