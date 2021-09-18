# import the necessary packages
import numpy as np
import cv2

MIN_CONF = 0.3
NMS_THRESH = 0.3
def yolov3(data,net,layers):
	height,width = data.shape[:2]
	scale = np.array([width,height,width,height])
	blob = cv2.dnn.blobFromImage(data, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(layers)
	results,confidences,bboxes = [],[],[]
	for output in outputs:
		for result in output:
			prob = result[5:]
			label = np.argmax(prob)	
			if label == 0 and prob[label] > MIN_CONF:
				loc = result[0:4] * scale
				(centerX, centerY, width, height) = loc.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				bboxes.append([x, y, int(width), int(height)])
				confidences.append(float(prob[label]))
	# Non Max Suppression to clear overlapped bounding boxes
	filterd_results = cv2.dnn.NMSBoxes(bboxes, confidences, MIN_CONF, NMS_THRESH)
	if len(filterd_results) > 0:
		for i in filterd_results.flatten():
			(x, y) = (bboxes[i][0], bboxes[i][1])
			(w, h) = (bboxes[i][2], bboxes[i][3])
			leftx = x
			lefty = y+h
			rightx = x+w
			righty = lefty
			results.append([leftx, lefty, rightx, righty])
	return results

