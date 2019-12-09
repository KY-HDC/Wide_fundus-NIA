import os
from datetime import datetime as T
import numpy as np
import cv2
import glob
import pandas as pd
import shutil
from joblib import Parallel, delayed

base_path  = 'dir'
step1_path = os.path.join(base_path, 'dir') 
step2_path = os.path.join(base_path, 'dir') 

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
def save_image(pImg, pFileName):
  cv2.imwrite(pFileName, pImg)
  print('Saved:', pFileName)
######################
def do_filtering(pFileName):
  
  img      = cv2.imread(pFileName)

  fileName = os.path.split(pFileName)[1]
  dst_name = fileName.replace('.jpg','_O.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(img, dst_path)

  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
  img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
  img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
  img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
  img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);
  dst_name = fileName.replace('.jpg','_SO.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(img_sobel, dst_path)
  
  Scharr = cv2.Sobel(img_gray, -1, 0, 1, ksize=-1)
  dst_name = fileName.replace('.jpg','_SC.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(Scharr, dst_path)
  
  gi       = cv2.GaussianBlur(img, (5, 5),0)
  dst_name = fileName.replace('.jpg','_G.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(gi, dst_path)

  mi       = cv2.medianBlur(img, 5)
  dst_name = fileName.replace('.jpg','_M.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(mi, dst_path)

  bi       = cv2.bilateralFilter(img,9,75,75)
  dst_name = fileName.replace('.jpg','_B.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(bi, dst_path)

  ker      = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  si       = cv2.filter2D(img, -1, ker)
  dst_name = fileName.replace('.jpg','_S.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(si, dst_path)

  imgcvted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  lc,ac,bc = cv2.split(imgcvted)
  clahe    = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(9, 9))
  cl       = clahe.apply(lc)
  merged   = cv2.merge((cl, ac, bc))
  hi = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
  dst_name = fileName.replace('.jpg','_H.JPG')
  dst_path = os.path.join(step2_path, dst_name)
  save_image(hi, dst_path)

######################
def run_main():
  query    = os.path.join(step1_path, '*.jpg')
  fileList = glob.glob(query)
  fileList.sort()
  
  make_subdir(step2_path)

  Parallel(n_jobs=-1)(delayed(do_filtering)
                      (fileName)
                      for fileName in fileList)
######################
if __name__ == '__main__':
  ST = T.now() 
  run_main()
  ET = T.now() 
  print('Elapsed Time =', ET-ST)
