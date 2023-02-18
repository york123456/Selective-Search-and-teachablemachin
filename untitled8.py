# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:13:20 2023

@author: b4100
"""

import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model("my_h5_model.h5", compile=False) 

# 讀取圖檔


# 要顯示的候選區域數量
numShowRects = 40

# 每次增加或減少顯示的候選區域數量
increment = 50
camera = cv2.VideoCapture(0)
while True:
  
  ret, image2 = camera.read()
  
  im=cv2.resize(image2,(300,300))
  
  
      
  # 建立 Selective Search 分割器
  ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
  # 設定要進行分割的圖形
  ss.setBaseImage(im)
    
  # 使用快速模式（精準度較差）
  ss.switchToSelectiveSearchFast()
    
  # 使用精準模式（速度較慢）
  # ss.switchToSelectiveSearchQuality()
  
  # 執行 Selective Search 分割
  rects = ss.process()
  
  print('候選區域總數量： {}'.format(len(rects)))




  # 複製一份原始影像
  imOut = im.copy()

  # 以迴圈處理每一個候選區域
  for i, rect in enumerate(rects):
      # 以方框標示候選區域
      if (i < numShowRects):
          crop_img=[]
          
          x, y, w, h = rect
          #cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
          
          # 裁切區域的 x 與 y 座標（左上角）
          x = x
          y = y
        
          # 裁切區域的長度與寬度
          w = w
          h = h
        
          # 裁切圖片
          crop_img = im[y:y+h, x:x+w]
          #cv2.imshow("Output2", crop_img)
          
          image = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA)
          # Show the image in a window
            
          # Make the image a numpy array and reshape it to the models input shape.
          image = np.asarray(image, dtype=np.float32).reshape(1, 28, 28, 3)
          # Normalize the image array
          image = (image / 255.0)
          # Have the model predict what the current image is. Model.predict
          # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
          # it is the first label and 80% sure its the second label.
          probabilities = model.predict(image)
          print(probabilities)
          
          if probabilities[0][0]>0.75 and probabilities[0][1]<0.025:#np.max(probabilities)>0.97: #and np.argmax(probabilities)==0 and # and np.min(probabilities)<0.05 :
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
          
          
          #cv2.waitKey(1)
          
          
      else:
          break

  # 顯示結果
  cv2.imshow("Output", imOut)

  # 讀取使用者所按下的鍵
  k = cv2.waitKey(1) & 0xFF

  # 若按下 m 鍵，則增加 numShowRects
  if k == 109:
      numShowRects += increment
  # 若按下 l 鍵，則減少 numShowRects
  elif k == 108 and numShowRects > increment:
      numShowRects -= increment
  # 若按下 q 鍵，則離開
  elif k == 113:
      break

# 關閉圖形顯示視窗
cv2.destroyAllWindows()