#%%
import cv2,os
from os import path
import detection,model
from detection import yolov3 
from model import load_model

def main(test_video):
	# check if the test video exists
	if not path.exists(test_video):
		return False
	# load video
	video = cv2.VideoCapture(test_video)
	# load model 
	net,layers = load_model()
	while True:
		# read data
		_,data = video.read()
		# get detection result
		result = yolov3(data,net,layers)
		# show the detected people 
		for _,axis in enumerate(result):
			(leftx, lefty, rightx, righty) = axis
			color = (0, 255, 0)
			cv2.line(data,(leftx,lefty),(rightx,righty),color,5)
		#show result
		cv2.imshow("Detection Results", data)
		key = cv2.waitKey(1) & 0xFF

test_video = "/Users/jiaruizhang/desktop/hackthenorth/video/test2.mp4"
main(test_video)
# %%
