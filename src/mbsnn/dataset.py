import torch 

from torch.utils.data import Dataset

import numpy as np

# Mandelbrot Dataset
class MandelbrotDataset(Dataset):
    def __init__(self, samples=1000, max_iter=20, max_magnitude=1e2, fix_c=False):
        self.samples = samples
        self.max_iter = max_iter
        self.max_mag = max_magnitude
        if fix_c:
            self.c = torch.complex(torch.rand(samples) * 3 - 2, torch.rand(samples) * 3 - 1.5)  # Complex c in [-2, 1] + [-1.5, 1.5]i
        else:
            self.c = None

    def generate_data(self, idx):
        while True:
            if self.c is None:
                c = torch.complex(torch.rand(1) * 3 - 2, torch.rand(1) * 3 - 1.5)  # Complex c in [-2, 1] + [-1.5, 1.5]i
            else:
                c = self.c[[idx]]
            #c = torch.tensor(-1.1 + 0.2j, dtype=torch.complex64).reshape(c.shape)
            z = torch.tensor(0 + 0j, dtype=torch.complex64)
            z_to_return = []
            t_to_return = []
            for t in range(1, self.max_iter + 1):
                z = z**2 + c
                
                if torch.isnan(z.real) or torch.isnan(z.imag) or torch.abs(z) > self.max_mag:
                    break
                
                z_to_return.append(z[0])
                t_to_return.append(t)
                
            # Randomly break the loop with probability t / max_iter, such that we have some incomplete sequences
            # this will favour C values that are either within the Mandelbrot set or at the boundary
            t_prob = t / self.max_iter
            
            repeat_prop = np.random.rand()
            if repeat_prop < t_prob:
                break
            
        if len(z_to_return) < (self.max_iter):
            z_to_return += [torch.tensor(torch.nan + torch.nan * 1j, dtype=torch.complex64)] * (self.max_iter - len(z_to_return))
            t_to_return += np.arange(len(t_to_return) + 1, self.max_iter + 1).tolist()
            
        z_to_return = torch.view_as_real(torch.stack(z_to_return))
        c = torch.view_as_real(c).repeat(self.max_iter, 1)
        return (torch.tensor(t_to_return, dtype=torch.float32).unsqueeze(-1), c, z_to_return)
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        t, c, z = self.generate_data(idx)
        
        return t, c, z
