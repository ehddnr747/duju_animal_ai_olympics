import numpy as np

class FrameBuffer(object):
    def __init__(self, step_size, channel_size, height, width):
        self.step_size = step_size
        self.channel_size = channel_size
        self.height = height
        self.width = width

        self.input_channel_size = step_size * channel_size

        self.buffer = np.zeros([self.input_channel_size, height, width])


    def dm_add(self, frame): # [height, width, channel] # black and white 3 channel input to 1 channel input
        frame = frame / 256.0
        frame = frame[:,:,[0]] * 0.2989 + frame[:,:,[1]] * 0.5870 + frame[:,:,[2]] * 0.1140
        frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
        self.buffer = np.concatenate([self.buffer[self.channel_size:], frame], axis=0)

        assert self.buffer.shape == (self.input_channel_size, self.height, self.width)



    # def dm_add(self, frame): # [height, width, channel]
    #     frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
    #     frame = (frame / 256.0)
    #     self.buffer = np.concatenate([self.buffer[self.channel_size:], frame], axis=0)
    #
    #     assert self.buffer.shape == (self.input_channel_size, self.height, self.width)


    def get_buffer(self):
        return self.buffer
        #[input_channel_size, height, width]