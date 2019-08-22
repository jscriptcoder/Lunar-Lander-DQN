import numpy as np

class SumTree:
    def __init__(self, mem_size):
        self.tree = np.zeros(2 * mem_size - 1)
        self.data = np.empty(mem_size, dtype=object)
        self.size = mem_size
        self.ptr  = 0
        self.n_elems = 0

    def update(self, idx, p):
        tree_idx = idx + self.size - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def store(self, p, data):
        if self.data[self.ptr] is None:
            self.n_elems += 1
        
        self.data[self.ptr] = data
        self.update(self.ptr, p)

        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0

    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1
        
        idx = ptr - (self.size - 1)
        return idx, self.tree[ptr], self.data[idx]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def max_p(self):
        return np.max(self.tree[-self.size:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.size:])