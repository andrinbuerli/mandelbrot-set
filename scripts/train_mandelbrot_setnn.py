import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import grad
import pytorch_lightning as pl
from torch.utils.data import Dataset
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler

import wandb

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.cosine_epochs = total_epochs - warmup_epochs
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr * warmup_factor for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing phase
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_factor = 0.5 * (1 + np.cos(torch.pi * cosine_epoch / self.cosine_epochs))
            return [self.base_lr * cosine_factor for _ in self.optimizer.param_groups]


def logscalemagnitude(z):
    #return z
    # log transform magnitude of z
    z_mag_log = torch.log1p(torch.abs(z) + 1e-6)
    # keep phase of z
    z_phase = torch.angle(z)
    # reconstruct z from log magnitude and phase
    z = torch.polar(z_mag_log, z_phase)
    return z

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

# Neural Network Model
class MandelbrotNN(pl.LightningModule):
    def __init__(
        self,
        num_hidden_layers=1,
        hidden_dim=64, 
        lr=1e-3, 
        max_iter=20, 
        warmup_epochs=5, 
        total_epochs=50, 
        pde_weight=1.0,
        output_dir="predictions/",
        max_batch_size=64,
    ):
        super().__init__()
        self.lr = lr
        self.pde_weight = pde_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_iter = max_iter
        self.max_batch_size = max_batch_size
        self.output_dir = output_dir
        self.model = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU()) 
                #nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()) 
                for _ in range(num_hidden_layers)
            ],
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, t, c):
        input_data = torch.cat([t, c], dim=-1)  # Combine t and real/imaginary parts of c
        z = self.model(input_data)
        return z

    def pde_loss(self, t, c):
        # Ensure c requires gradients for dz/dc computation
        c.requires_grad_(True)
        # Time derivative: d/dt
        t.requires_grad_(True)
        z_pred_t = self.forward(t, c)
        z_pred_t_complex = torch.view_as_complex(z_pred_t.contiguous())
        tminus1 = t - 1  # Shift time backward by 1
        z_pred_tminus1 = self.forward(tminus1, c)
        z_pred_tminus1_complex = torch.view_as_complex(z_pred_tminus1.contiguous())
        
        # Compute dz/dt
        dz_dt = grad(z_pred_t_complex, t, torch.ones_like(z_pred_t_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]
        dz_dtminus1 = grad(z_pred_tminus1_complex, tminus1, torch.ones_like(z_pred_tminus1_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]

        # Residual for time derivative in the complex domain
        residual_t_complex = dz_dt.squeeze() - 2 * z_pred_tminus1_complex * dz_dtminus1.squeeze()
        
        # Gradients with respect to c (real and imaginary parts separately)
        dz_dc_t = grad(z_pred_t_complex, c, torch.ones_like(z_pred_t_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]
        dz_dc_t_complex = torch.view_as_complex(dz_dc_t.contiguous())
        
        dz_dc_tminus1 = grad(z_pred_tminus1_complex, c, torch.ones_like(z_pred_tminus1_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]
        dz_dc_tminus1_complex = torch.view_as_complex(dz_dc_tminus1.contiguous())

        # Compute residual for d/dc
        residual_c_complex = dz_dc_t_complex - (2 * z_pred_tminus1_complex * dz_dc_tminus1_complex + 1)

        # Compute the final loss
        mean_residual_t = torch.mean(residual_t_complex.abs())
        mean_residual_c = torch.mean(residual_c_complex.abs())
        
        return mean_residual_t, mean_residual_c
    
    def r2_score_complex(self, z_true, z_pred):
        """
        Calculate the R² score.
        """
        ss_res = torch.sum((torch.view_as_complex(z_true) - torch.view_as_complex(z_pred)).abs() ** 2)
        ss_tot = torch.sum((torch.view_as_complex(z_true) - torch.mean(torch.view_as_complex(z_true))).abs() ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()  # Convert to Python float for logging

    def training_step(self, batch, batch_idx):
        t, c, z_true = batch
        
        t = t.view(-1, 1)
        c = c.view(-1, 2)
        z_true = z_true.view(-1, 2)
        
        nan_mask = torch.isnan(z_true).any(-1)
        t = t[~nan_mask]
        c = c[~nan_mask]
        z_true = z_true[~nan_mask]
        
        if z_true.shape[0] > self.max_batch_size:
            # Randomly sample a subset of the batch
            idx = torch.randperm(z_true.shape[0])[:self.max_batch_size]
            t = t[idx]
            c = c[idx]
            z_true = z_true[idx]
        
        z_pred = self.forward(t, c)
        
        data_loss = self.data_loss(z_true, z_pred)
        
        r2 = self.r2_score_complex(z_true=z_true, z_pred=z_pred)
        
        # only compute PDE loss on not escaped points
        residual_t, residual_c = self.pde_loss(t, c)
        pde_loss = residual_t + residual_c
        loss = data_loss + self.pde_weight * pde_loss

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/data_loss", data_loss, on_epoch=True)
        self.log("train/pde_loss", pde_loss, on_epoch=True)
        self.log("train/residual_t", residual_t, on_epoch=True)
        self.log("train/residual_c", residual_c, on_epoch=True)
        self.log("train/r2", r2, on_epoch=True, prog_bar=True)
        return loss

    def data_loss(self, z_true, z_pred):
        data_loss = (torch.view_as_complex(z_true) - torch.view_as_complex(z_pred)).abs().mean()
        return data_loss

    def validation_step(self, batch, batch_idx):
        t, c, z_true = batch
        
        t = t.view(-1, 1)
        c = c.view(-1, 2)
        z_true = z_true.view(-1, 2)
        
        nan_mask = torch.isnan(z_true).any(-1)
        t = t[~nan_mask]
        c = c[~nan_mask]
        z_true = z_true[~nan_mask]
        
        if z_true.shape[0] > self.max_batch_size:
            # Randomly sample a subset of the batch
            idx = torch.randperm(z_true.shape[0])[:self.max_batch_size]
            t = t[idx]
            c = c[idx]
            z_true = z_true[idx]
        
        with torch.enable_grad():
            z_pred = self.forward(t, c)
            residual_t, residual_c = self.pde_loss(t, c)
        
        pde_loss = residual_t + residual_c
        data_loss = self.data_loss(z_true, z_pred)
        
        r2 = self.r2_score_complex(z_true=z_true, z_pred=z_pred)
        
        loss = data_loss + self.pde_weight * pde_loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/data_loss", data_loss, on_epoch=True, prog_bar=True)
        self.log("val/pde_loss", pde_loss, on_epoch=True, prog_bar=True)
        self.log("val/residual_t", residual_t, on_epoch=True, prog_bar=True)
        self.log("val/residual_c", residual_c, on_epoch=True, prog_bar=True)
        self.log("val/r2", r2, on_epoch=True, prog_bar=True)

        if batch_idx == 0:  # Plot predictions vs real on first batch
            self.plot_predictions(c, self.max_iter//2, save_dir=f"{self.output_dir}_inter/")
            self.plot_predictions(c, self.max_iter, save_dir=f"{self.output_dir}/")
            self.plot_predictions(c, self.max_iter * 2, save_dir=f"{self.output_dir}_extrapol/")
            self.plot_mandelbrot_set(save_path=f"{self.output_dir}_inter/mandelbrot_plot.png", iterations=self.max_iter // 2)
            self.plot_mandelbrot_set(save_path=f"{self.output_dir}/mandelbrot_plot.png", iterations=self.max_iter)
            self.plot_mandelbrot_set(save_path=f"{self.output_dir}_extrapol/mandelbrot_plot.png", iterations=self.max_iter * 2)
            
            # plot zoomed at xmin, xmax, ymin, ymax = -1.0, -0.5, -0.5, 0.0
            self.plot_mandelbrot_set(
                save_path=f"{self.output_dir}_inter/mandelbrot_plot_zoomed.png", 
                iterations=self.max_iter // 2,
                real_min=-1.0, real_max=-0.5,
                imag_min=-0.5, imag_max=0.0)
            self.plot_mandelbrot_set(
                save_path=f"{self.output_dir}/mandelbrot_plot_zoomed.png",
                iterations=self.max_iter,
                real_min=-1.0, real_max=-0.5,
                imag_min=-0.5, imag_max=0.0)
            self.plot_mandelbrot_set(
                save_path=f"{self.output_dir}_extrapol/mandelbrot_plot_zoomed.png",
                iterations=self.max_iter * 2,
                real_min=-1.0, real_max=-0.5,
                imag_min=-0.5, imag_max=0.0)
    
    def plot_mandelbrot_set(self, real_min=-2.0, real_max=1.0, imag_min=-1.5, imag_max=1.5, resolution=500, iterations=20, save_path='mandelbrot_plot.png'):
        device = next(self.parameters()).device
        # Create grid of complex points c
        real = torch.linspace(real_min, real_max, resolution, device=device)
        imag = torch.linspace(imag_min, imag_max, resolution, device=device)
        re, im = torch.meshgrid(real, imag, indexing='ij')  
        c_grid = torch.stack([re, im], dim=-1).reshape(-1, 2)  # shape: (resolution*resolution, 2)
        
        # Set t = 2 * self.max_iter
        t_val = torch.full((c_grid.shape[0], 1), iterations, dtype=torch.float32, device=device)

        # Run forward pass
        with torch.no_grad():
            z_pred = self.forward(t_val, c_grid)
        
        # Convert predictions to complex and compute magnitude
        z_complex = torch.view_as_complex(z_pred)
        magnitude = torch.abs(z_complex).reshape(resolution, resolution).cpu().numpy()

        # Plot the log of the magnitude to highlight differences
        plt.figure(figsize=(8, 8))
        plt.imshow(np.where(((~np.isnan(magnitude.T)) & (magnitude.T < 2.0)), magnitude.T, 2.0), 
                extent=[real_min, real_max, imag_min, imag_max], 
                origin='lower', 
                cmap='hot',)

        plt.colorbar(label='Clipped Magnitude')
        plt.xlabel('Real Axis')
        plt.ylabel('Imag Axis')
        plt.title(f'Mandelbrot Set Visualization at t={iterations}')

        # Save and close figure
        plt.savefig(save_path, dpi=150)
        wandb.save(save_path)
        plt.close()
        
    def plot_predictions(self, c_batch, max_iter, save_dir="predictions/"):
        os.makedirs(save_dir, exist_ok=True)
        
        # compute unique c values
        c_batch = c_batch.unique(dim=0)[:8]
                
        for idx, c in enumerate(c_batch):
            c_complex = torch.view_as_complex(c)
            z_real = [0 + 0j]  # Ground truth values
            z_nn = [0 + 0j]    # NN predicted values
            z = torch.tensor(0 + 0j, dtype=torch.complex64)

            # Generate ground truth and NN predictions
            for t in range(1, max_iter):
                z = z**2 + c_complex
                
                if torch.isnan(z.real) or torch.isnan(z.imag) or torch.abs(z) > 1e2:
                    break
                
                z_real.append(logscalemagnitude(z).cpu().numpy())
                t_tensor = torch.tensor([t], dtype=torch.float32, device=c.device).unsqueeze(0)  
                z_pred = self(t_tensor, c[None])[0]
                z_nn.append(logscalemagnitude(torch.view_as_complex(z_pred)).cpu().item())

            # Convert to numpy arrays for plotting
            z_real = np.array(z_real)
            z_pred = np.array(z_nn)

            # Create a color map for the time steps
            norm = mcolors.Normalize(vmin=0, vmax=t)
            cmap = cm.viridis
            colors = [cmap(norm(tt)) for tt in range(t + 1)]

            # Create a new figure with two subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            # Subplot 1: Ground Truth with Time Gradient
            for i in range(len(z_real) - 1):
                axs[0].plot(
                    [np.real(z_real[i]), np.real(z_real[i + 1])],
                    [np.imag(z_real[i]), np.imag(z_real[i + 1])],
                    color=colors[i],
                )
            axs[0].set_title(f"Ground Truth Trajectory for $c = {c_complex.item():.2f}$")
            axs[0].set_xlabel("Real")
            axs[0].set_ylabel("Imaginary")
            axs[0].grid()

            # Add colorbar to the first subplot
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=axs[0], orientation="vertical", pad=0.1)
            cbar.set_label("Time Step")

            # Subplot 2: NN Prediction with Time Gradient
            for i in range(len(z_pred) - 1):
                axs[1].plot(
                    [np.real(z_pred[i]), np.real(z_pred[i + 1])],
                    [np.imag(z_pred[i]), np.imag(z_pred[i + 1])],
                    color=colors[i],
                )
            axs[1].set_title(f"NN Prediction Trajectory for $c = {c_complex.item():.2f}$")
            axs[1].set_xlabel("Real")
            axs[1].set_ylabel("Imaginary")
            axs[1].grid()
            
            
            for i in range(len(z_pred) - 1):
                axs[2].scatter(
                    [np.real(z_pred[i]), np.real(z_pred[i + 1])],
                    [np.imag(z_pred[i]), np.imag(z_pred[i + 1])],
                    color=colors[i],
                    marker='o',
                )
                axs[2].scatter(
                    [np.real(z_real[i]), np.real(z_real[i + 1])],
                    [np.imag(z_real[i]), np.imag(z_real[i + 1])],
                    color=colors[i],
                    marker='x',
                )

            # Add colorbar to the second subplot
            cbar = plt.colorbar(sm, ax=axs[1], orientation="vertical", pad=0.1)
            cbar.set_label("Time Step")

            # Save the plot with a unique name
            plt.tight_layout()
            file_name = f"{save_dir}prediction_{idx}.png"
            plt.savefig(file_name)
            wandb.save(file_name)
            #self.logger.log_image(key=f"{save_dir}Prediction Trajectory for $c = {c_complex.item():.2f}$", images=[fig], step=self.global_step)
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.total_epochs,
            base_lr=self.lr,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


# Main Training Loop
if __name__ == "__main__":
    import os
    #os.environ["WANDB_MODE"]="offline"
    dataset = MandelbrotDataset(samples=100_000, max_iter=20)
    val_dataset = MandelbrotDataset(samples=10_000, max_iter=20, fix_c=True)

    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)

    model = MandelbrotNN(
        hidden_dim=1024,
        num_hidden_layers=6, 
        lr=1e-4, 
        max_iter=20,
        warmup_epochs=1, 
        total_epochs=1000,
        pde_weight=0.1,
        max_batch_size=2048,
        output_dir="predictions",)
    
    # add mlflow logger
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    wandb_logger = pl.loggers.WandbLogger(project="mandelbrot-set-nn", entity="andrinburli", log_model=True)
    
    #wandb_logger.watch(model, log="all", log_freq=1000)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints',
        filename='mandelbrot-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_epochs=1000,
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[lr_monitor, pl.callbacks.ModelSummary(max_depth=3), checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
