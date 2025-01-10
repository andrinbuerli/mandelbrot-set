import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import pytorch_lightning as pl
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
import wandb

from mbsnn.utils import WarmupCosineScheduler, logscalemagnitude

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
            figs = self.plot_predictions(c, self.max_iter//2)
            self.save_figs(figs, save_dir=f"{self.output_dir}_inter/")
            
            figs = self.plot_predictions(c, self.max_iter)
            self.save_figs(figs, save_dir=f"{self.output_dir}/")
            
            figs = self.plot_predictions(c, self.max_iter * 2)
            self.save_figs(figs, save_dir=f"{self.output_dir}_extrapol/")
            
            fig = self.plot_mandelbrot_set(iterations=self.max_iter // 2)
            self.save_figs([fig], save_dir=f"{self.output_dir}_inter/", file_name="mandelbrot_plot")
            
            fig = self.plot_mandelbrot_set(iterations=self.max_iter)
            self.save_figs([fig], save_dir=f"{self.output_dir}/", file_name="mandelbrot_plot")
            
            fig = self.plot_mandelbrot_set(iterations=self.max_iter * 2)
            self.save_figs([fig], save_dir=f"{self.output_dir}_extrapol/", file_name="mandelbrot_plot")
            
            # plot zoomed at xmin, xmax, ymin, ymax = -1.0, -0.5, -0.5, 0.0
            zoom_args = dict(real_min=-1.0, real_max=-0.5, imag_min=-0.5, imag_max=0.0)
            
            fig = self.plot_mandelbrot_set(iterations=self.max_iter // 2, **zoom_args)
            self.save_figs([fig], save_dir=f"{self.output_dir}_inter/", file_name="mandelbrot_plot_zoomed")
            
            fig = self.plot_mandelbrot_set(iterations=self.max_iter, **zoom_args)
            self.save_figs([fig], save_dir=f"{self.output_dir}/", file_name="mandelbrot_plot_zoomed")
            
            fig = self.plot_mandelbrot_set(iterations=self.max_iter * 2, **zoom_args)
            self.save_figs([fig], save_dir=f"{self.output_dir}_extrapol/", file_name="mandelbrot_plot_zoomed")

    @staticmethod
    def save_figs(figs, save_dir, file_name="prediction"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for ii, fig in enumerate(figs):
                # Save the plot with a unique name
            fig.tight_layout()
            fn = f"{save_dir}{file_name}_{ii}.png"
            plt.savefig(fn)
            wandb.save(fn)
            #self.logger.log_image(key=f"{save_dir}Prediction Trajectory for $c = {c_complex.item():.2f}$", images=[fig], step=self.global_step)
            plt.close(fig)
    
    def plot_mandelbrot_set(self, real_min=-2.0, real_max=1.0, imag_min=-1.5, imag_max=1.5, resolution=500, iterations=20):
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
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.where(((~np.isnan(magnitude.T)) & (magnitude.T < 2.0)), magnitude.T, 2.0), 
                extent=[real_min, real_max, imag_min, imag_max], 
                origin='lower', 
                cmap='hot',)

        plt.colorbar(label='Clipped Magnitude')
        plt.xlabel('Real Axis')
        plt.ylabel('Imag Axis')
        plt.title(f'Mandelbrot Set Visualization at t={iterations}')

        return fig
        
    def plot_predictions(self, c_batch, max_iter):
        # compute unique c values
        c_batch = c_batch.unique(dim=0)[:8]
                
        figs = []
        for c in c_batch:
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
            
            figs.append(fig)
            
        return figs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.total_epochs,
            base_lr=self.lr,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

