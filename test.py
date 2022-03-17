import numpy as np
import cv2 as cv


def main():
    #inti detector
    cfg = r'F:\TUhh\Thesis\Endpoint-Detection/yolov4-tiny-custom.cfg'
    weight = r'F:\TUhh\Thesis\Endpoint-Detection/yolov4-tiny-custom_5000.weights'
    net = cv.dnn.readNetFromDarknet(cfg, weight)
    classes = ['Corrosion']

    cap = cv.VideoCapture(1)
    flag = 0
    ctr = 0
    while True:
        _, img = cap.read()
        img = cv.resize(img, (416, 416))
        hight,width,_ = img.shape
        blob = cv.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

        net.setInput(blob)
        output_layers_name = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_name)

        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3]* hight)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)


        indexes = cv.dnn.NMSBoxes(boxes,confidences,.5,.4)
        font = cv.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size =(len(boxes),3))
        if len(indexes)==0 and flag==1:
            ctr = ctr+1
            if ctr >5:
                color = (255,0,0)
                cv.putText(img,"defect removed", (15,15),font,2,color,2)
                if ctr >10:
                    ctr = 0
                    flag = 0
    
        if  len(indexes)>0:
            flag = 1
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv.putText(img,label + " " + confidence, (x,y),font,2,color,2)

        cv.imshow('img', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
    #print(cv.__version__)
