from pyueye import ueye
import numpy as np


class IDSAccess:
    def __init__(self):

        self.hcam = ueye.HIDS(0)
        self.ret = ueye.is_InitCamera(self.hcam, None)

        # set color mode
        self.ret = ueye.is_SetColorMode(self.hcam, ueye.IS_CM_BGR8_PACKED)

        # set region of interest
        self.width = 960
        self.height = 960
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(0)
        rect_aoi.s32Y = ueye.int(0)
        rect_aoi.s32Width = ueye.int(self.width)
        rect_aoi.s32Height = ueye.int(self.height)
        ueye.is_AOI(self.hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        # allocate memory
        self.mem_ptr = ueye.c_mem_p()
        mem_id = ueye.int()
        self.bitspixel = 24  # for colormode = IS_CM_BGR8_PACKED
        self.ret = ueye.is_AllocImageMem(self.hcam, self.width, self.height, self.bitspixel, self.mem_ptr, mem_id)

        # set active memory region
        self.ret = ueye.is_SetImageMem(self.hcam, self.mem_ptr, mem_id)

        # continuous capture to memory
        ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
        print(f"CaptureVideo returns {ret}")

        # get data from camera and display
        self.lineinc = self.width * int((self.bitspixel + 7) / 8)

    def read(self):
        img = ueye.get_data(self.mem_ptr, self.width, self.height, self.bitspixel, self.lineinc, copy=False)
        img = np.reshape(img, (self.height, self.width, 3))
        out = (True, img)
        return out

    def release(self):
        self.ret = ueye.is_StopLiveVideo(self.hcam, ueye.IS_FORCE_VIDEO_STOP)
        self.ret = ueye.is_ExitCamera(self.hcam)
