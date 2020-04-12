# COMPARISON OF VEHICLE AND PEDESTRIAN DETECTION USING YOLOv3 MODEL TRAINED ON COCO AND KITTI DATASET

The comparison of YOLOv3 model trained on COCO dataset and KITTI dataset is inferred to generate the following,

![](/images/2.png)

1. YOLOv3 COCO prediction confidence vs object labels
2. YOLOV3 KITTI prediction confidence vs object labels
3. Prediction time comparison
4. Bounding box x1-y1 comparison for the model predictions 
5. Bounding box x2-y2 comparison for the model predictions

The flow of python code implementation is as follows, 

![](/images/1.png)

## Required Python Libraries and Files

1. Python 
2. OpenCV 
3. OpenCV-contrib
4. Matplotlib
5. Time 
6. Glob
7. Openpyxl
8. Weights file of the yolov3 models
Download yolov3.weights for COCO dataset from this link and add it to the yolo_coco directory,
[click here](https://pjreddie.com/darknet/yolo/)
9. Uncompress yolo_kitti.7z

## Code Execution 
`cd Yolov3 COCO vs KITTI` 

`python yolo_coco_kitti.py` 
