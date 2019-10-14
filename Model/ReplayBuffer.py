import numpy as np

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, buffer_size):  # Image state will be handled at ImageBuffer
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.count = 0
        self.ptr = 0

        self.s_buffer = np.zeros([buffer_size, state_dim])
        self.a_buffer = np.zeros([buffer_size, action_dim])
        self.r_buffer = np.zeros([buffer_size, 1])
        self.t_buffer = np.zeros([buffer_size, 1])
        self.s2_buffer = np.zeros([buffer_size, state_dim])

    def size(self):
        return self.count

    def add(self, s, a, r, t, s2):
        assert s.shape == (self.state_dim,)
        assert a.shape == (self.action_dim,)
        assert r.shape == (1,)
        assert t.shape == (1,)
        assert s2.shape == (self.state_dim,)

        self.s_buffer[self.ptr] = s
        self.a_buffer[self.ptr] = a
        self.r_buffer[self.ptr] = r
        self.t_buffer[self.ptr] = t
        self.s2_buffer[self.ptr] = s2

        self.count = min(self.count + 1, self.buffer_size)
        self.ptr = (self.ptr + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        batch_size = min(self.count, batch_size)

        sample_idx = np.random.choice(self.count, batch_size, replace=False)

        return self.s_buffer[sample_idx], self.a_buffer[sample_idx], self.r_buffer[sample_idx], self.t_buffer[
            sample_idx], self.s2_buffer[sample_idx]

