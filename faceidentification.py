# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:56:42 2020

@author: amit
"""
import numpy as np
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}

with open ("label.pickle", "rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
    
    
    
cap = cv2.VideoCapture(0)
while True:
        # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w,h) in faces:
        print (x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # recognizer
        # deep learnmodel predict
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45:# and conf<= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x, y), font, 1, color, stroke, cv2.LINE_AA)
             
        
        img_item = "Myimage.png"
        cv2.imwrite(img_item, roi_gray)
        # draw rectangle
        color = (255,0,0)
        stroke = 5
        end_chord_x = x+w
        end_chord_y = y+h
        cv2.rectangle(frame, (x,y), (end_chord_x, end_chord_y), color, stroke)
        
  
    # Display the resulting frame
    cv2.imshow('frame',frame)
      # cv2.imshow('blue',blue)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

