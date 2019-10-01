import numpy as np

class ImageBuffer(object):
    def __init__(self, height, width, stepsize):
        self.height = height
        self.width = width
        self.stepsize = stepsize
        self.buffer = np.zeros([stepsize,height,width])

    # [height, width, channel]
    def dm_add(self, frame):
        frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
        self.buffer = np.concatenate([self.buffer[1:], frame], axis=0)

        assert self.buffer.shape == (self.stepzie, self.height, self.width)

    def get_buffer(self):
        return self.buffer
    #[stepsize, height, width]