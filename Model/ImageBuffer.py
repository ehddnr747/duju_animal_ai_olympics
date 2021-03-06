import numpy as np

class ImageBuffer(object):

    # idx starts from 1. The first image will have idx 1.
    # When the buffer get full with batch size 100, then there will be 100 iamges and the current idx will be 100.

    def __init__(self, height, width, stepsize, channel_size, buffer_size):
        self.height = height
        self.width = width
        self.stepsize = stepsize
        self.channel_size = channel_size
        self.step_channelsize = stepsize * channel_size

        self.buffer = []
        self.buffer_size = buffer_size
        self.count = 0

        assert self.stepsize <= self.buffer_size

        self.full_count = 0


    # # [height, width, channel]
    # def dm_add(self, frame):
    #     frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
    #     frame = (frame / 128.0) - 1.0
    #
    #     if self.count < self.buffer_size:
    #         self.count += 1
    #         self.buffer.append(frame)
    #     else:
    #         self.buffer.pop(0)
    #         self.buffer.append(frame)
    #         self.full_count += 1

    # [height, width, channel]
    def dm_add_gray(self, frame):
        frame = frame / 256.0
        frame = frame[:, :, [0]] * 0.2989 + frame[:, :, [1]] * 0.5870 + frame[:, :, [2]] * 0.1140
        frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0]) # [1, height, width]

        assert frame.shape == (1, self.height, self.width)

        if self.count < self.buffer_size:
            self.count += 1
            self.buffer.append(frame)
        else:
            self.buffer.pop(0)
            self.buffer.append(frame)
            self.full_count += 1

    def animal_add(self, frame):

        #store as np.ubyte

        frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0]) * 256
        frame = np.array(frame, dtype=np.ubyte)

        assert frame.shape == (self.channel_size, self.height, self.width)

        if self.count < self.buffer_size:
            self.count += 1
            self.buffer.append(frame)
        else:
            self.buffer.pop(0)
            self.buffer.append(frame)
            self.full_count += 1


    def get_state(self, idx):
        assert idx > self.full_count
        assert idx <= self.count + self.full_count

        if self.count < self.buffer_size:
            temp_idx = idx
        else:
            temp_idx = idx - self.full_count

        return_array = np.concatenate(self.buffer[temp_idx - self.stepsize: temp_idx], axis=0)
        # because image idx starts from 1 and list idx starts from 0

        assert return_array.shape == (self.step_channelsize, self.height, self.width)

        return return_array / np.array(256, dtype=np.float32)
        #[stepsize, height, width]

    def get_current_image(self, idx):

        assert idx > self.full_count
        assert idx <= self.count + self.full_count

        if self.count < self.buffer_size:
            temp_idx = idx
        else:
            temp_idx = idx - self.full_count

        return_array = self.buffer[temp_idx-1]
        # because image idx starts from 1 and list idx starts from 0

        assert return_array.shape == (self.channel_size, self.height, self.width)

        return return_array / np.array(256, dtype=np.float32)

    def get_state_and_next(self, idx):

        assert idx > 0 and idx < self.count + self.full_count

        return self.get_state(idx), self.get_state(idx+1)

    def get_current_index(self):
        return self.count + self.full_count