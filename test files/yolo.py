'''
Computer Vision Project - Venkatagiri Ramesh & Pranav Krishna Prasad
'''
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt


'''YOLOv3_COCO - Initialization'''
labels_coco_path = "yolo_coco/coco.names"
labels_coco = open(labels_coco_path).read().strip().split("\n")

np.random.seed(42)
colors_coco = np.random.randint(0, 255, size=(len(labels_coco), 3),
	dtype="uint8")

weights_coco = "yolo_coco/yolov3.weights"
config_coco = "yolo_coco/yolov3.cfg"

net_coco = cv2.dnn.readNetFromDarknet(config_coco, weights_coco)


'''YOLOv3_KIITI- Initialization'''
labels_kitti_path= "yolo_kitti/kitti.names"
labels_kitti = open(labels_kitti_path).read().strip().split("\n")

np.random.seed(42)
colors_kitti = np.random.randint(0, 255, size=(len(labels_kitti), 3),
	dtype="uint8")

weights_kitti = "yolo_kitti/yolov3-kitti.weights"
config_kitti = "yolo_kitti/yolov3-kitti.cfg"

net_kitti = cv2.dnn.readNetFromDarknet(config_kitti, weights_kitti)


'''Image input pipline'''
img_coco =cv2.imread('test/test.jpg')
img_kitti =cv2.imread('test/test.jpg')

'''Image yolov3 coco processing'''
(H, W) = img_coco.shape[:2]

ln = net_coco.getLayerNames()
ln = [ln[i[0] - 1] for i in net_coco.getUnconnectedOutLayers()]

blob_coco = cv2.dnn.blobFromImage(img_coco, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net_coco.setInput(blob_coco)
start_coco = time.time()
layerOutputs = net_coco.forward(ln)
end_coco = time.time()
predtime_coco = end_coco - start_coco
print("Frame Prediction Time - YOLOv3_COCO dataset : {:.6f} seconds".format(predtime_coco))

boxes_coco = []
confidences_coco = []
classIDs_coco = []


for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > 0.5:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes_coco.append([x, y, int(width), int(height)])
			confidences_coco.append(float(confidence))
			classIDs_coco.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes_coco, confidences_coco, 0.5,0.3)


if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes_coco[i][0], boxes_coco[i][1])
		(w, h) = (boxes_coco[i][2], boxes_coco[i][3])

		color = [int(c) for c in colors_coco[classIDs_coco[i]]]
		cv2.rectangle(img_coco, (x, y), (x + w, y + h), color, 2)
		f = 0.02
		rw = 0.08
		dist = w * f/rw
		print(dist)

		text = "{}: {:.4f}".format(labels_coco[classIDs_coco[i]], confidences_coco[i])
		cv2.putText(img_coco, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


'''Image yolov3 kitti processing'''
(H, W) = img_kitti.shape[:2]
ln = net_kitti.getLayerNames()
ln = [ln[i[0] - 1] for i in net_kitti.getUnconnectedOutLayers()]

blob_kitti = cv2.dnn.blobFromImage(img_kitti, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net_kitti.setInput(blob_kitti)
start_kitti = time.time()
layerOutputs = net_kitti.forward(ln)
end_kitti = time.time()
predtime_kitti = end_kitti - start_kitti
print("Frame Prediction Time - YOLOv3_KIITI dataset : {:.6f} seconds".format(predtime_kitti))

boxes_kitti = []
confidences_kitti = []
classIDs_kitti = []


for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > 0.5:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes_kitti.append([x, y, int(width), int(height)])
			confidences_kitti.append(float(confidence))
			classIDs_kitti.append(classID)

idxs_kitti = cv2.dnn.NMSBoxes(boxes_kitti, confidences_kitti, 0.5,0.3)


if len(idxs_kitti) > 0:
	for i in idxs_kitti.flatten():
		(x, y) = (boxes_kitti[i][0], boxes_kitti[i][1])
		(w, h) = (boxes_kitti[i][2], boxes_kitti[i][3])

		color = [int(c) for c in colors_kitti[classIDs_kitti[i]]]
		cv2.rectangle(img_kitti, (x, y), (x + w, y + h), color, 2)
		f = 0.02
		rw = 0.08
		dist = w * f/rw
		print(dist)

		text = "{}: {:.4f}".format(labels_kitti[classIDs_kitti[i]], confidences_kitti[i])
		cv2.putText(img_kitti, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

plt.subplot(1,2,1)
plt.imshow(img_coco)
plt.subplot(1, 2, 2)
plt.imshow(img_kitti)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
