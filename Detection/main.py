from Detector import Detector


CFPATH = r'G:\yolov4-tiny-custom.cfg'
WEIGHTPATH = r'G:\yolov4-tiny-custom_best.weights'
CLASS = ['Mask', 'No_Mask']

def detectFromImage(file):
    detect = Detector( weight=WEIGHTPATH, cfg=CFPATH, height=416, width=416, classes=CLASS)
    detect.imageDetection(file)


def detectFromVideo(file):
    print("in_development")



def detectFromCam(isIDS = False):
    detect = Detector(weight=WEIGHTPATH, cfg=CFPATH, height=416, width=416, classes=CLASS)
    detect.camDetection(isIDS)



if __name__ == '__main__':

    filePath = r'G:\archive\images\maksssksksss21.png'

    detectFromImage(filePath)
    #detectFromVideo(filePath)
    #detectFromCam(False)




