import cv2
import numpy as np 
import glob
import random
import pandas as pd 

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_training.cfg')

classes = ['cumen']

images = glob.glob('train_images/*.jpg')

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# random.shuffle(images)

pred = []
ids = []

for img_path in images:
	id_ = img_path.split('\\')[-1].split('.')[0]
	ids.append(id_)

	img = cv2.imread(img_path)
	img = cv2.resize(img, (800,800))

	height, width, channels = img.shape

	blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), (0,0,0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)

	class_ids = []
	confidences = []
	boxes = []


	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:

				centerx = int(detection[0] * width)
				centery = int(detection[1] * height)

				w = int(detection[2] * width)
				h = int(detection[3] * height)
				x = int(centerx - w/2)
				y = int(centery - h/2)

				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)#bboxes, scores, threshold, non-max suppression

	font = cv2.FONT_HERSHEY_PLAIN
	count = 0
	for i in range(len(boxes)):
		if i in indexes:
			count += 1
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[class_ids[i]]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
	
	cv2.putText(img, str(count), (width//2, height//2), font, 3, (0,0,255), 3)
	pred.append(count)

	cv2.imshow('img', img)
	key = cv2.waitKey(0)

	if key == ord(' '):
		continue
	elif key == ord('q'):
		break

cv2.destroyAllWindows()

# output = pd.DataFrame({'Id': ids, 'Predicted': pred})
# output.to_csv('my_submission.csv', index=False)