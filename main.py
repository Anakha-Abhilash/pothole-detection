import cv2
from math import ceil
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
import winsound


video_capture = cv2.VideoCapture(0)

model = load_model('model/Roadss_model.h5')

font = cv2.FONT_HERSHEY_DUPLEX

while True:
	ret,frame = video_capture.read()

	X = cv2.resize(frame,(256,256))
	X = np.array(X)
	X = np.expand_dims(X, axis=0)
	y_pred = np.round(model.predict(X))

	res=y_pred[0][0]

	if res==1:
		result="Plain Road"
	else:
		result="Pothole Road"
		winsound.Beep(1000,500)
		
	cv2.putText(frame,result,(10,50),font,1,(255,255,255),2)
	cv2.imshow('Video', frame)
	k = cv2.waitKey(1) & 0xFF


	#Press 'Escape' to exit web cam 
	if k == 27:
		break
	

video_capture.release()
cv2.destroyAllWindows()
