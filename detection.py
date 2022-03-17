from pyueye import ueye
import numpy as np
import cv2


def main():
    # init camera
    cfg = r'F:\TUhh\Thesis\Endpoint-Detection/yolov4-tiny-custom.cfg'
    weight = r'F:\TUhh\Thesis\Endpoint-Detection/yolov4-tiny-custom_5000.weights'
    net = cv2.dnn.readNetFromDarknet(cfg, weight)
    classes = ['Corrosion']
    flag = 0
    ctr = 0

    hcam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(hcam, None)
    print(f"initCamera returns {ret}")

    # set color mode
    ret = ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)
    #ret = ueye.is_SetColorMode(hcam, ueye.IS_COLORMODE_CBYCRY)
    print(f"SetColorMode IS_CM_BGR8_PACKED returns {ret}")

    # set region of interest
    wt = 1024
    ht = 1024
    rect_aoi = ueye.IS_RECT()
    rect_aoi.s32X = ueye.int(0)
    rect_aoi.s32Y = ueye.int(0)
    rect_aoi.s32Width = ueye.int(wt)
    rect_aoi.s32Height = ueye.int(ht)
    ueye.is_AOI(hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
    print(f"AOI IS_AOI_IMAGE_SET_AOI returns {ret}")

    # allocate memory
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()
    bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
    ret = ueye.is_AllocImageMem(hcam, wt, ht, bitspixel, mem_ptr, mem_id)

    # set active memory region
    ret = ueye.is_SetImageMem(hcam, mem_ptr, mem_id)




    # continuous capture to memory
    ret = ueye.is_CaptureVideo(hcam, ueye.IS_DONT_WAIT)


    # get data from camera and display
    lineinc = wt * int((bitspixel + 7) / 8)
    while True:
        img = ueye.get_data(mem_ptr, wt, ht, bitspixel, lineinc, copy=False)
        img = np.reshape(img, (wt, ht , 3))
        img = cv2.resize(img, (416, 416))
        hight, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        output_layers_name = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_name)

        boxes = []
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
                    h = int(detection[3] * hight)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) == 0 and flag == 1:
            ctr = ctr + 1
            if ctr > 5:
                color = (255, 0, 0)
                cv2.putText(img, "defect removed", (15, 15), font, 2, color, 2)
                if ctr > 10:
                    ctr = 0
                    flag = 0

        if len(indexes) > 0:
            flag = 1
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y), font, 2, color, 2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # cleanup
    ret = ueye.is_StopLiveVideo(hcam, ueye.IS_FORCE_VIDEO_STOP)
    print(f"StopLiveVideo returns {ret}")
    ret = ueye.is_ExitCamera(hcam)
    print(f"ExitCamera returns {ret}")


if __name__ == '__main__':
    #main()
    print(cv2.__version__)