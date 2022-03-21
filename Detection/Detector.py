import cv2 as cv
import numpy as np
from IDSAccess import IDSAccess


class Detector:
    def __init__(self, weight, cfg, height, width, classes):

        self.height = height
        self.width = width
        self.classes = classes
        self.net = cv.dnn.readNetFromDarknet(cfg, weight)
        self.boxes = []
        self.confidences = []
        self.class_ids = []

    def imageDetection(self, path):

        img = cv.imread(path)
        img = cv.resize(img, (self.height, self.width))
        blob = cv.dnn.blobFromImage(img, 1/255, (self.height, self.width), (0, 0, 0), swapRB=True, crop=False)


        self.net.setInput(blob)
        outputLayerName = self.net.getUnconnectedOutLayersNames()
        layerOutputs = self.net.forward(outputLayerName)

        indexes = self.getBoundingbox(layerOutputs)
        self.showOutput(indexes=indexes, img=img)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def camDetection(self, isIDS=False):

        if isIDS:
            cap = IDSAccess()

        else:
            cap = cv.VideoCapture(0)

        while True:
            _, img = cap.read()
            img = cv.resize(img, (self.width, self.height))
            blob = cv.dnn.blobFromImage(img, 1 / 255, (self.height, self.width), (0, 0, 0), swapRB=True, crop=False)

            self.net.setInput(blob)
            outputLayerName = self.net.getUnconnectedOutLayersNames()
            layerOutputs = self.net.forward(outputLayerName)

            indexes = self.getBoundingbox(layerOutputs)
            self.showOutput(indexes=indexes, img=img)
            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()



    def getBoundingbox(self, layerOutputs):

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = detection[4]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.boxes.append([x, y, w, h])
                    self.confidences.append((float(confidence)))
                    self.class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(self.boxes, self.confidences, .5, .4)
        return indexes

    def showOutput(self, indexes, img):
        font = cv.FONT_HERSHEY_PLAIN
        color = (0, 0, 255)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = self.boxes[i]
                label = str(self.classes[self.class_ids[i]])
                confidence = str(round(self.confidences[i], 2))
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label + " " + confidence, (x, y), font, 2, color, 2)

        cv.imshow('img', img)
        self.boxes = []
        self.confidences = []
        self.class_ids = []
