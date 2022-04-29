import time


class FPS:

    def __init__(self):
        self.prevFrame = 0
        self.newFrame = 0

    def getFPS(self):
        self.newFrame = time.time()
        fps = 1/(self.newFrame-self.prevFrame)
        self.prevFrame = self.newFrame
        fps = int(fps)
        fps = str(fps)

        return fps
