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
        pde_weight_warmup_epochs=0,
        output_dir="predictions/",
        max_batch_size=64,
        max_magnitude=1e2,
        time_token_dim=None,
        compute_dz_dt=True,
        compute_dz_dc=True,
        log_scale=False,
        use_clipped_nans=False,
    ):
        super().__init__()
        self.lr = lr
        self.use_clipped_nans = use_clipped_nans
        self.log_scale = log_scale
        self.compute_dz_dt = compute_dz_dt
        self.compute_dz_dc = compute_dz_dc
        self.max_magnitude = max_magnitude
        self.pde_weight = pde_weight
        self.pde_weight_warmup_epochs = pde_weight_warmup_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_iter = max_iter
        self.max_batch_size = max_batch_size
        self.output_dir = output_dir
        self.model = nn.Sequential(
            nn.Linear(3 if time_token_dim is None else time_token_dim, hidden_dim),
            nn.GELU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU()) 
                #nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()) 
                for _ in range(num_hidden_layers)
            ],
            nn.Linear(hidden_dim, 2)
        )
        
        if time_token_dim is not None:
            self.time_token_embedding = nn.Embedding(max_iter*2 + 1, time_token_dim)
            self.c_embedding = nn.Sequential(
                nn.Linear(2, time_token_dim),
                nn.GELU(),
                nn.Linear(time_token_dim, time_token_dim),
            )
        else:
            self.time_token_embedding = None
            self.c_embedding = None

    def forward(self, t, c):
        if self.time_token_embedding is not None:
            assert t.max() < self.time_token_embedding.num_embeddings, f"Time token {t.max()} exceeds max_iter {self.time_token_embedding.num_embeddings}"
            t = self.time_token_embedding(t.long())[..., 0, :]
            c = self.c_embedding(c)
            input_data = t + c
        else:
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
        
        
        if self.compute_dz_dt:
            # Compute dz/dt
            dz_dt = grad(z_pred_t_complex, t, torch.ones_like(z_pred_t_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]
            dz_dtminus1 = grad(z_pred_tminus1_complex, tminus1, torch.ones_like(z_pred_tminus1_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]

            # Residual for time derivative in the complex domain
            residual_t_complex = dz_dt.squeeze() - 2 * z_pred_tminus1_complex * dz_dtminus1.squeeze()
        else:
            residual_t_complex = torch.zeros_like(z_pred_t_complex)
        
        if self.compute_dz_dc:
            # Gradients with respect to c (real and imaginary parts separately)
            dz_dc_t = grad(z_pred_t_complex, c, torch.ones_like(z_pred_t_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]
            dz_dc_t_complex = torch.view_as_complex(dz_dc_t.contiguous())
            
            dz_dc_tminus1 = grad(z_pred_tminus1_complex, c, torch.ones_like(z_pred_tminus1_complex), retain_graph=True, create_graph=True, allow_unused=True)[0]
            dz_dc_tminus1_complex = torch.view_as_complex(dz_dc_tminus1.contiguous())

            # Compute residual for d/dc
            residual_c_complex = dz_dc_t_complex - (2 * z_pred_tminus1_complex * dz_dc_tminus1_complex + 1)
        else:
            residual_c_complex = torch.zeros_like(z_pred_t_complex)

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
        
        if self.use_clipped_nans:
            z_true[nan_mask] = self.max_magnitude
            z_true_masked = z_true
            t_masked = t
            c_masked = c
        else:
            t_masked = t[~nan_mask]
            c_masked = c[~nan_mask]
            z_true_masked = z_true[~nan_mask]
            
        if z_true_masked.shape[0] > self.max_batch_size:
            # Randomly sample a subset of the batch
            idx = torch.randperm(z_true_masked.shape[0])[:self.max_batch_size]
            t_masked = t_masked[idx]
            c_masked = c_masked[idx]
            z_true_masked = z_true_masked[idx]
        z_pred = self.forward(t_masked, c_masked)
        data_loss = self.data_loss(z_true_masked, z_pred)
        
        r2 = self.r2_score_complex(z_true=z_true_masked, z_pred=z_pred)
        
        # only compute PDE loss on not escaped points
        residual_t, residual_c = self.pde_loss(t[~nan_mask], c[~nan_mask])
        pde_loss = residual_t + residual_c
        loss = data_loss + self.get_pde_weight() * pde_loss

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/data_loss", data_loss, on_epoch=True)
        self.log("train/pde_loss", pde_loss, on_epoch=True)
        self.log("train/pde_weight", self.get_pde_weight(), on_epoch=True)
        self.log("train/residual_t", residual_t, on_epoch=True)
        self.log("train/residual_c", residual_c, on_epoch=True)
        self.log("train/r2", r2, on_epoch=True, prog_bar=True)
        self.log("train/batch_size", z_true_masked.shape[0], on_epoch=True, prog_bar=False)
        return loss

    def data_loss(self, z_true, z_pred):
        data_loss = (torch.view_as_complex(z_true) - torch.view_as_complex(z_pred)).abs().mean()
        return data_loss

    def get_pde_weight(self):
        # linear warmup for PDE weight
        if self.current_epoch < self.pde_weight_warmup_epochs:
            return self.pde_weight * self.current_epoch / self.pde_weight_warmup_epochs
        return self.pde_weight

    def validation_step(self, batch, batch_idx):
        t, c, z_true = batch
        
        t = t.view(-1, 1)
        c = c.view(-1, 2)
        z_true = z_true.view(-1, 2)
        
        nan_mask = torch.isnan(z_true).any(-1)
        if self.use_clipped_nans:
            z_true[nan_mask] = self.max_magnitude
            z_true_masked = z_true
            t_masked = t
            c_masked = c
        else:
            t_masked = t[~nan_mask]
            c_masked = c[~nan_mask]
            z_true_masked = z_true[~nan_mask]
            
        if z_true_masked.shape[0] > self.max_batch_size:
            # Randomly sample a subset of the batch
            idx = torch.randperm(z_true_masked.shape[0])[:self.max_batch_size]
            t_masked = t_masked[idx]
            c_masked = c_masked[idx]
            z_true_masked = z_true_masked[idx]
        
        with torch.enable_grad():
            z_pred = self.forward(t_masked, c_masked)
            residual_t, residual_c = self.pde_loss(t[nan_mask], c[nan_mask])
        
        pde_loss = residual_t + residual_c
        data_loss = self.data_loss(z_true_masked, z_pred)
        
        r2 = self.r2_score_complex(z_true=z_true_masked, z_pred=z_pred)
        
        loss = data_loss + self.get_pde_weight() * pde_loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/data_loss", data_loss, on_epoch=True, prog_bar=True)
        self.log("val/pde_loss", pde_loss, on_epoch=True, prog_bar=True)
        self.log("val/pde_weight", self.get_pde_weight(), on_epoch=True, prog_bar=True)
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
            
            # Plot Mandelbrot set without dzdc
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter // 2, dzdc=False, output_dir=f"{self.output_dir}_inter", suffix="")
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter, dzdc=False, output_dir=f"{self.output_dir}", suffix="")
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter * 2, dzdc=False, output_dir=f"{self.output_dir}_extrapol", suffix="")
            
            # Plot Mandelbrot set with dzdc
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter // 2, dzdc=True, output_dir=f"{self.output_dir}_inter", suffix="_dzdc")
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter, dzdc=True, output_dir=f"{self.output_dir}", suffix="_dzdc")
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter * 2, dzdc=True, output_dir=f"{self.output_dir}_extrapol", suffix="_dzdc")
            
            # Plot Mandelbrot set with dzdc_gt
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter // 2, dzdc=False, dzdc_gt=True, output_dir=f"{self.output_dir}_inter", suffix="_dzdc_gt")
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter, dzdc=False, dzdc_gt=True, output_dir=f"{self.output_dir}", suffix="_dzdc_gt")
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter * 2, dzdc=False, dzdc_gt=True, output_dir=f"{self.output_dir}_extrapol", suffix="_dzdc_gt")
            
            # Plot zoomed Mandelbrot set without dzdc
            zoom_args = dict(real_min=-1.0, real_max=-0.5, imag_min=-0.5, imag_max=0.0)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter // 2, dzdc=False, output_dir=f"{self.output_dir}_inter", suffix="_zoomed", **zoom_args)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter, dzdc=False, output_dir=f"{self.output_dir}", suffix="_zoomed", **zoom_args)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter * 2, dzdc=False, output_dir=f"{self.output_dir}_extrapol", suffix="_zoomed", **zoom_args)
            
            # Plot zoomed Mandelbrot set with dzdc
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter // 2, dzdc=True, output_dir=f"{self.output_dir}_inter", suffix="_zoomed_dzdc", **zoom_args)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter, dzdc=True, output_dir=f"{self.output_dir}", suffix="_zoomed_dzdc", **zoom_args)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter * 2, dzdc=True, output_dir=f"{self.output_dir}_extrapol", suffix="_zoomed_dzdc", **zoom_args)

            # Plot zoomed Mandelbrot set with dzdc_gt
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter // 2, dzdc=False, dzdc_gt=True, output_dir=f"{self.output_dir}_inter", suffix="_zoomed_dzdc_gt", **zoom_args)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter, dzdc=False, dzdc_gt=True, output_dir=f"{self.output_dir}", suffix="_zoomed_dzdc_gt", **zoom_args)
            self.plot_and_save_mandelbrot_set(iterations=self.max_iter * 2, dzdc=False, dzdc_gt=True, output_dir=f"{self.output_dir}_extrapol", suffix="_zoomed_dzdc_gt", **zoom_args)

    def plot_and_save_mandelbrot_set(self, iterations, dzdc, suffix, output_dir, **kwargs):
        fig = self.plot_mandelbrot_set(iterations=iterations, dzdc=dzdc, **kwargs)
        save_dir = f"{output_dir}/"
        self.save_figs([fig], save_dir=save_dir, file_name=f"mandelbrot_plot{suffix}")

    @staticmethod
    def save_figs(figs, save_dir, file_name="prediction"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for ii, fig in enumerate(figs):
            # Save the plot with a unique name
            fig.tight_layout()
            fn = f"{save_dir}{file_name}_{ii}.png"
            fig.savefig(fn)
            wandb.save(fn)
            #self.logger.log_image(key=f"{save_dir}Prediction Trajectory for $c = {c_complex.item():.2f}$", images=[fig], step=self.global_step)
            plt.close(fig)
    
    def plot_mandelbrot_set(self, real_min=-2.0, real_max=1.0, imag_min=-1.5, imag_max=1.5, resolution=500, iterations=20, dzdc=False, dzdc_gt=False):
        device = next(self.parameters()).device
        # Create grid of complex points c
        real = torch.linspace(real_min, real_max, resolution, device=device)
        imag = torch.linspace(imag_min, imag_max, resolution, device=device)
        re, im = torch.meshgrid(real, imag, indexing='ij')  
        c_grid = torch.stack([re, im], dim=-1).reshape(-1, 2)  # shape: (resolution*resolution, 2)
        
        # Set t = 2 * self.max_iter
        t_val = torch.full((c_grid.shape[0], 1), iterations, dtype=torch.float32, device=device)

        # Run forward pass
        if not dzdc and not dzdc_gt:
            with torch.no_grad():
                z_pred = self.forward(t_val, c_grid)
            # Convert predictions to complex and compute magnitude
            z_complex = torch.view_as_complex(z_pred)
            magnitude = torch.abs(z_complex).reshape(resolution, resolution).cpu().numpy()
        else:
            if dzdc:
                with torch.enable_grad():
                    c_grid.requires_grad_(True)
                    dz_dc = []
                    for i in range(0, t_val.shape[0], self.max_batch_size):
                        t_batch = t_val[i:i + self.max_batch_size]
                        c_batch = c_grid[i:i + self.max_batch_size]
                        z_pred_batch = self.forward(t_batch, c_batch)
                        dz_dc_batch = grad(z_pred_batch, c_batch, torch.ones_like(z_pred_batch), create_graph=True, allow_unused=True)[0]
                        dz_dc.append(dz_dc_batch.detach().cpu())
                dz_dc = torch.view_as_complex(torch.cat(dz_dc, dim=0))
                magnitude = torch.abs(dz_dc).reshape(resolution, resolution).cpu().numpy()
            elif dzdc_gt:
                with torch.enable_grad():
                    c_grid.requires_grad_(True)
                    z_pred_tminus1 = []
                    dz_dc_tminus1 = []
                    for i in range(0, t_val.shape[0], self.max_batch_size):
                        t_batch = t_val[i:i + self.max_batch_size] - 1
                        c_batch = c_grid[i:i + self.max_batch_size]
                        z_pred_tminus1_batch = self.forward(t_batch, c_batch)
                        z_pred_tminus1.append(z_pred_tminus1_batch.detach().cpu())
                        dz_dc_tminus1_batch = grad(z_pred_tminus1_batch, c_batch, torch.ones_like(z_pred_tminus1_batch), create_graph=True, allow_unused=True)[0]
                        dz_dc_tminus1.append(dz_dc_tminus1_batch.detach().cpu())
                dz_dc_tminus1 = torch.view_as_complex(torch.cat(dz_dc_tminus1, dim=0))
                z_pred_tminus1 = torch.view_as_complex(torch.cat(z_pred_tminus1, dim=0))
                dz_dc_gt = 2 * z_pred_tminus1 * dz_dc_tminus1 + 1
                magnitude = torch.abs(dz_dc_gt).reshape(resolution, resolution).cpu().numpy()
        

        # Plot the log of the magnitude to highlight differences
        fig = plt.figure(figsize=(8, 8))
        threshold = 2.0 if not self.log_scale else np.log1p(2.0)
        plt.imshow(
            np.where(
                    ((~np.isnan(magnitude.T)) & (magnitude.T < threshold)), 
                    magnitude.T, threshold
                ) if not dzdc and not dzdc_gt else np.log(magnitude.T), 
                extent=[real_min, real_max, imag_min, imag_max], 
                origin='lower', 
                cmap='hot',)

        plt.colorbar(label='Clipped Magnitude' if not dzdc and not dzdc_gt else 'Log Gradient Magnitude')
        plt.xlabel('Real Axis')
        plt.ylabel('Imag Axis')
        dzdcstr = 'with $\\frac{\\partial z}{\\partial c}$' if dzdc else ''
        dzdcstr = 'with $\\frac{\\partial z}{\\partial c}$ GT' if dzdc_gt else dzdcstr
        plt.title(f"Mandelbrot Set Visualization at t={iterations} {dzdcstr}")

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
                
                if torch.isnan(z.real) or torch.isnan(z.imag) or torch.abs(z) > self.max_magnitude:
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

