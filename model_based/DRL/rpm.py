import torch
import numpy as np

class rpm(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.current_size = 0
        
        self.s0 = None
        self.a = None
        self.r = None
        self.s1 = None
        self.d = None

    def _init_buffers(self, s0, a, r, s1, d):
        dtype_s = s0.dtype
        if s0.max() > 255: 
            pass 
        
        self.s0 = torch.empty((self.buffer_size,) + s0.shape, dtype=s0.dtype, device='cpu', pin_memory=True)
        self.a = torch.empty((self.buffer_size,) + a.shape, dtype=a.dtype, device='cpu', pin_memory=True)
        self.r = torch.empty((self.buffer_size,) + r.shape, dtype=r.dtype, device='cpu', pin_memory=True)
        self.s1 = torch.empty((self.buffer_size,) + s1.shape, dtype=s1.dtype, device='cpu', pin_memory=True)
        self.d = torch.empty((self.buffer_size,) + d.shape, dtype=d.dtype, device='cpu', pin_memory=True)
        
        total_bytes = (self.s0.element_size() * self.s0.nelement() + 
                       self.a.element_size() * self.a.nelement() +
                       self.r.element_size() * self.r.nelement() +
                       self.s1.element_size() * self.s1.nelement() +
                       self.d.element_size() * self.d.nelement())
        print(f"Replay Buffer initialized on CPU. Size: {total_bytes / 1024**2:.2f} MB", flush=True)

    def append(self, transition):
        s0, a, r, s1, d = transition
        
        if self.s0 is None:
            self._init_buffers(s0, a, r, s1, d)
            
        self.s0[self.index] = s0
        self.a[self.index] = a
        self.r[self.index] = r
        self.s1[self.index] = s1
        self.d[self.index] = d
        
        self.index = (self.index + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def append_batch(self, s0, a, r, s1, d):
        batch_size = s0.shape[0]
        
        if self.s0 is None:
            self._init_buffers(s0[0], a[0], r[0], s1[0], d[0])
            
        if self.index + batch_size <= self.buffer_size:
            self.s0[self.index:self.index + batch_size] = s0
            self.a[self.index:self.index + batch_size] = a
            self.r[self.index:self.index + batch_size] = r
            self.s1[self.index:self.index + batch_size] = s1
            self.d[self.index:self.index + batch_size] = d
        else:
            part1 = self.buffer_size - self.index
            part2 = batch_size - part1
            
            self.s0[self.index:self.buffer_size] = s0[:part1]
            self.a[self.index:self.buffer_size] = a[:part1]
            self.r[self.index:self.buffer_size] = r[:part1]
            self.s1[self.index:self.buffer_size] = s1[:part1]
            self.d[self.index:self.buffer_size] = d[:part1]
            
            self.s0[:part2] = s0[part1:]
            self.a[:part2] = a[part1:]
            self.r[:part2] = r[part1:]
            self.s1[:part2] = s1[part1:]
            self.d[:part2] = d[part1:]
            
        self.index = (self.index + batch_size) % self.buffer_size
        self.current_size = min(self.current_size + batch_size, self.buffer_size)

    def size(self):
        return self.current_size

    def sample_batch(self, batch_size, device):
        if self.current_size == 0:
            return None
            
        indices = np.random.randint(0, self.current_size, size=batch_size)
        
        s0 = self.s0[indices].to(device, non_blocking=True)
        a = self.a[indices].to(device, non_blocking=True)
        r = self.r[indices].to(device, non_blocking=True)
        s1 = self.s1[indices].to(device, non_blocking=True)
        d = self.d[indices].to(device, non_blocking=True)
        
        return s0, a, r, s1, d
