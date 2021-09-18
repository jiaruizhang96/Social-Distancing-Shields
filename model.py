#%%
import cv2
import os
from os import path

def load_model():
	# model directory path predefined
	model_path = os.path.dirname(__file__)+"/model"
	for files in os.listdir(model_path):
		suffix = files.split(".")[-1]
		if suffix == "cfg":
			config = os.path.join(model_path,files)
		if suffix == "weights":
			weights = os.path.join(model_path,files)
	# load model
	net = cv2.dnn.readNetFromDarknet(config, weights)
	# determine only the *output* layer names that we need from YOLO
	temp = net.getLayerNames()
	layers = []
	for i in net.getUnconnectedOutLayers():
		# index = i[0] - 1
		layers.append(temp[i[0]-1])
	return net,layers
