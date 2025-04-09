import torch 

from torch.utils.data import Dataset

import numpy as np

from mbsnn.utils import logscalemagnitude

# Mandelbrot Dataset
class MandelbrotDataset(Dataset):
    def __init__(
        self, 
        samples=1000,
        max_iter=20, 
        max_magnitude=1e2, 
        fix_c=False,
        biased_sampling=False,
        real_range=(-2, 1),
        imag_range=(-1.5, 1.5),
        log_scale=False):
        
        self.samples = samples
        self.log_scale = log_scale
        self.max_iter = max_iter
        self.max_mag = max_magnitude
        self.real_range = real_range
        self.imag_range = imag_range
        if fix_c:
            # sample from real and imaginary range
            self.c = torch.complex(
                torch.tensor(np.random.uniform(real_range[0], real_range[1], samples), dtype=torch.float32), 
                torch.tensor(np.random.uniform(imag_range[0], imag_range[1], samples), dtype=torch.float)
            )
        else:
            self.c = None
            
        self.biased_sampling = biased_sampling

    def generate_data(self, idx):
        while True:
            if self.c is None:
                c = torch.complex(
                    torch.tensor(np.random.uniform(self.real_range[0], self.real_range[1]), dtype=torch.float32), 
                    torch.tensor(np.random.uniform(self.imag_range[0], self.imag_range[1]), dtype=torch.float)
                ).reshape(1)
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
            if repeat_prop < t_prob or self.c is not None or not self.biased_sampling:
                break
            
        if len(z_to_return) < (self.max_iter):
            z_to_return += [torch.tensor(torch.nan + torch.nan * 1j, dtype=torch.complex64)] * (self.max_iter - len(z_to_return))
            t_to_return += np.arange(len(t_to_return) + 1, self.max_iter + 1).tolist()
            
        z_to_return = torch.stack(z_to_return)
        if self.log_scale:
            z_to_return = logscalemagnitude(z_to_return)
        z_to_return = torch.view_as_real(z_to_return)
        c = torch.view_as_real(c).repeat(self.max_iter, 1)
        return (torch.tensor(t_to_return, dtype=torch.float32).unsqueeze(-1), c, z_to_return)
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        t, c, z = self.generate_data(idx)
        
        return t, c, z
