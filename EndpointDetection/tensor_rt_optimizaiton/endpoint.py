import cv2 as cv
import numpy as np
from IDSAccess import IDSAccess


import os
import time
import argparse

import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO






def camDetection(isIDS=False, isCuda = False):

    if isCuda:
        #self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        #self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        None


    if isIDS:
        cap = IDSAccess()

    else:
        cap = cv.VideoCapture(0)

    trt_yolo = TrtYOLO('yolov4-tiny-custom')
    font = cv.FONT_HERSHEY_PLAIN
    color = (0, 0, 255)
    fps = 0.0
    tic = time.time()
    flag = 0
    ctr = 0
    temp_boxes = None
    while True:
        _, img = cap.read()
        img = cv.resize(img, (640, 640))

        boxes, confs, clss = trt_yolo.detect(img, 0.3)
        
        if len(boxes) > 0:
            flag = 1
            ctr = 0
            temp_boxes = boxes
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        if len(boxes) == 0 and flag == 1:
            ctr = ctr + 1
            if ctr < 60: # to maintain lag of 3 seconds from defect removal to message display
                for box in temp_boxes:
                    x_min, y_min, x_max, y_max = box
                    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            if ctr > 60:
                cv.putText(img, "Defect Removed", (50, 50), font, 2, color, 2)
            if ctr > 100:
                flag = 0
                ctr = 0
                

        cv.imshow('Detection',img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)

        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        print("FPS :"+ str(int(fps)))

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':

   camDetection(isIDS=True, isCuda=False)