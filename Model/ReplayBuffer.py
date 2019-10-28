import numpy as np
import operator

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

# For Segment Tree
class Segment_Tree(object):
    def __init__(self, capacity, operation, init_value):
        assert (capacity > 0 and capacity & (capacity - 1) == 0), "capacity must be positive and a power of 2."

        self.capacity = capacity
        self.operation = operation
        self.tree = np.ones(2 * capacity, dtype = np.float32) * init_value

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumTree(Segment_Tree):
    def __init__(self, capacity):
        super(SumTree, self).__init__(capacity=capacity,
                                      operation=operator.add,
                                      init_value=0.0)

    def sum(self):
        return self.tree[1]

    def retrieve(self, upperbound, count):
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.tree[left] < upperbound:
                upperbound -= self.tree[left]
                idx = right
            else:
                idx = left

        return min(idx - self.capacity, count - 1)

class MinTree(Segment_Tree):
    def __init__(self, capacity):
        super(MinTree, self).__init__(capacity = capacity,
                                     operation = min,
                                     init_value = float("inf"))
    def min(self):
        return self.tree[1]


class PERBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, alpha, initial_beta, max_step):
        super(PERBuffer, self).__init__(state_dim=state_dim,
                                        action_dim=action_dim,
                                        buffer_size=buffer_size)

        assert alpha > 0.0 and initial_beta > 0.0

        self.alpha = alpha
        self.initial_beta = initial_beta
        self.beta = self.initial_beta

        self.current_step = 0
        self.max_step = max_step

        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sumTree = SumTree(tree_capacity)
        self.minTree = MinTree(tree_capacity)

        self.tree_ptr = 0

        self.max_priority = 1.0  # This is containg alpha inside

    def store(self, s, a, r, t, s2):
        self.add(s, a, r, t, s2)
        self.sumTree[self.tree_ptr] = self.max_priority
        self.minTree[self.tree_ptr] = self.max_priority
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def update_beta(self):
        self.current_step += 1
        self.beta = self.initial_beta + (1 - self.initial_beta) * self.current_step / self.max_step

    def sample_batch(self, batch_size):

        assert self.count >= batch_size

        indices = self._sample_proportional(batch_size)

        s = self.s_buffer[indices]
        a = self.a_buffer[indices]
        r = self.r_buffer[indices]
        t = self.t_buffer[indices]
        s2 = self.s2_buffer[indices]
        weights = self._calculate_weights(indices)

        return s, a, r, t, s2, indices, weights

    def update_priorities(self, indices, td_errors):
        assert len(indices) == len(td_errors)

        td_errors = td_errors + 1e-5

        for idx, td in zip(indices, td_errors):
            self.sumTree[idx] = td ** self.alpha
            self.minTree[idx] = td ** self.alpha

        self.max_priority = max(self.max_priority, max(td_errors) ** self.alpha)

    def _sample_proportional(self, batch_size):

        indices = []
        p_sum = self.sumTree.sum()
        upper_bounds = np.random.uniform(0.0, p_sum, batch_size)

        for ub in upper_bounds:
            indices.append(self.sumTree.retrieve(ub, self.count))

        return indices

    def _calculate_weights(self, indices):

        tree_min = self.minTree.min()

        weights = (tree_min / np.array([self.sumTree[idx] for idx in indices])) ** self.beta

        return weights