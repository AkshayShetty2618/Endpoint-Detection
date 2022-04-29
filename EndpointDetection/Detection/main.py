from Detector import Detector


CFPATH = r'/home/ifp/project/tensorrt_demos/yolo/960_tiny/yolov4-tiny-custom.cfg'
WEIGHTPATH = r'/home/ifp/project/tensorrt_demos/yolo/960_tiny/yolov4-tiny-custom.weights'
CLASS = ['Corrosion']

def detectFromImage(file):
    detect = Detector( weight=WEIGHTPATH, cfg=CFPATH, height=416, width=416, classes=CLASS)
    detect.imageDetection(file)


def detectFromVideo(file):
    print("in_development")



def detectFromCam(isIDS = False, isCUDA = False):
    detect = Detector(weight=WEIGHTPATH, cfg=CFPATH, height=960, width=960, classes=CLASS)
    detect.camDetection(isIDS=isIDS, isCuda=isCUDA)



if __name__ == '__main__':

    #filePath = r'G:\archive\images\maksssksksss21.png'

    #detectFromImage(filePath)
    #detectFromVideo(filePath)
    detectFromCam(isIDS=True,isCUDA=True)
 
