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
    def __init__(self, samples=1000, max_iter=20, fix_c=False):
        self.samples = samples
        self.max_iter = max_iter
        if fix_c:
            self.c = torch.complex(torch.rand(samples) * 3 - 2, torch.rand(samples) * 3 - 1.5)  # Complex c in [-2, 1] + [-1.5, 1.5]i
        else:
            self.c = None

    def generate_data(self, idx):
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
            
            if torch.isnan(z.real) or torch.isnan(z.imag) or torch.abs(z) > 2:
                break
            
            z_to_return.append(z[0])
            t_to_return.append(t)
            
            
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
    ):
        super().__init__()
        self.lr = lr
        self.pde_weight = pde_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_iter = max_iter
        self.output_dir = output_dir
        self.model = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) 
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
        t_shifted = t - 1  # Shift time backward by 1
        z_pred_shifted = self.forward(t_shifted, c)
        
        # Compute dz/dt
        dz_dt = grad(z_pred_t, t, torch.ones_like(z_pred_t), retain_graph=True, create_graph=True, allow_unused=True)[0]
        dz_shifted_dt = grad(z_pred_shifted, t_shifted, torch.ones_like(z_pred_shifted), retain_graph=True, create_graph=True, allow_unused=True)[0]

        # Convert z_pred and c to complex tensors
        z_pred_complex = torch.view_as_complex(z_pred_shifted)

        # Residual for time derivative in the complex domain
        residual_t = dz_dt.squeeze() - 2 * z_pred_complex * dz_shifted_dt.squeeze()
        
        # Gradients with respect to c (real and imaginary parts separately)
        dz_dc_t = grad(z_pred_t, c, torch.ones_like(z_pred_t), retain_graph=True, create_graph=True, allow_unused=True)[0]
        dz_dc_real_t = dz_dc_t[..., 0]
        dz_dc_imag_t = dz_dc_t[..., 1]

        dz_dc_t_minus_1 = grad(z_pred_shifted, c, torch.ones_like(z_pred_shifted), retain_graph=True, create_graph=True, allow_unused=True)[0]
        dz_dc_real_t_minus_1 = dz_dc_t_minus_1[..., 0]
        dz_dc_imag_t_minus_1 = dz_dc_t_minus_1[..., 1]

        if dz_dc_real_t is None or dz_dc_imag_t is None or dz_dc_real_t_minus_1 is None or dz_dc_imag_t_minus_1 is None:
            raise RuntimeError("Gradient computation for dz/dc failed. Ensure c is used in the graph.")

        # Combine real and imaginary gradients for the residual
        dz_dc_t = torch.complex(dz_dc_real_t, dz_dc_imag_t)
        dz_dc_t_minus_1 = torch.complex(dz_dc_real_t_minus_1, dz_dc_imag_t_minus_1)

        # Compute residual for d/dc
        residual_c = dz_dc_t - (2 * z_pred_complex * dz_dc_t_minus_1 + 1)

        # Compute the final loss
        mean_residual_t = torch.mean(residual_t.abs())
        mean_residual_c = torch.mean(residual_c.abs())
        
        return mean_residual_t, mean_residual_c
    
    def r2_score_complex(self, z_true, z_pred):
        """
        Calculate the R² score.
        """
        ss_res = torch.sum((torch.view_as_complex(z_true) - torch.view_as_complex(z_pred)) ** 2)
        ss_tot = torch.sum((torch.view_as_complex(z_true) - torch.mean(torch.view_as_complex(z_true))) ** 2)
        r2 = 1 - ss_res.abs() / ss_tot.abs()
        return r2.item()  # Convert to Python float for logging


    def training_step(self, batch, batch_idx):
        t, c, z_true = batch
        z_pred = self.forward(t, c)
        
        nan_mask = torch.isnan(z_true).any(-1)
        z_true_imputed = self._get_imputed_z_true(z_true, nan_mask.unsqueeze(-1))

        data_loss = nn.MSELoss()(z_pred, z_true_imputed)
        
        r2_unescaped = self.r2_score_complex(z_true_imputed[~nan_mask], z_pred[~nan_mask])
        r2_imputed = self.r2_score_complex(z_true_imputed, z_pred)
        r2_escaped = self.r2_score_complex(z_true_imputed[nan_mask], z_pred[nan_mask])
        
        # only compute PDE loss on not escaped points
        residual_t, residual_c = self.pde_loss(t[~nan_mask], c[~nan_mask])
        pde_loss = residual_t + residual_c
        loss = data_loss + self.pde_weight * pde_loss

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/data_loss", data_loss, on_epoch=True)
        self.log("train/pde_loss", pde_loss, on_epoch=True)
        self.log("train/residual_t", residual_t, on_epoch=True)
        self.log("train/residual_c", residual_c, on_epoch=True)
        self.log("train/r2_unescaped", r2_unescaped, on_epoch=True, prog_bar=True)
        self.log("train/r2_imputed", r2_imputed, on_epoch=True, prog_bar=True)
        self.log("train/r2_escaped", r2_escaped, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        t, c, z_true = batch
        nan_mask = torch.isnan(z_true).any(-1)
        z_true_imputed = self._get_imputed_z_true(z_true, nan_mask.unsqueeze(-1))
        with torch.enable_grad():
            z_pred = self.forward(t, c)
            residual_t, residual_c = self.pde_loss(t[~nan_mask], c[~nan_mask])
        
        pde_loss = residual_t + residual_c
        data_loss = nn.MSELoss()(z_pred, z_true_imputed)
        
        r2_unescaped = self.r2_score_complex(z_true_imputed[~nan_mask], z_pred[~nan_mask])
        r2_imputed = self.r2_score_complex(z_true_imputed, z_pred)
        r2_escaped = self.r2_score_complex(z_true_imputed[nan_mask], z_pred[nan_mask])
        
        loss = data_loss + self.pde_weight * pde_loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/data_loss", data_loss, on_epoch=True, prog_bar=True)
        self.log("val/pde_loss", pde_loss, on_epoch=True, prog_bar=True)
        self.log("val/residual_t", residual_t, on_epoch=True, prog_bar=True)
        self.log("val/residual_c", residual_c, on_epoch=True, prog_bar=True)
        self.log("val/r2_unescaped", r2_unescaped, on_epoch=True, prog_bar=True)
        self.log("val/r2_imputed", r2_imputed, on_epoch=True, prog_bar=True)
        self.log("val/r2_escaped", r2_escaped, on_epoch=True, prog_bar=True)

        if batch_idx == 0:  # Plot predictions vs real on first batch
            self.plot_predictions(c, self.max_iter//2, save_dir=f"{self.output_dir}_inter/")
            self.plot_predictions(c, self.max_iter, save_dir=f"{self.output_dir}/")
            self.plot_predictions(c, self.max_iter * 2, save_dir=f"{self.output_dir}_extrapol/")
            self.plot_mandelbrot_set(save_path=f"{self.output_dir}_inter/mandelbrot_plot.png", iterations=self.max_iter // 2)
            self.plot_mandelbrot_set(save_path=f"{self.output_dir}/mandelbrot_plot.png", iterations=self.max_iter)
            self.plot_mandelbrot_set(save_path=f"{self.output_dir}_extrapol/mandelbrot_plot.png", iterations=self.max_iter * 2)


    def _get_imputed_z_true(self, z_true, nan_mask):
        escape_time = torch.argmax(nan_mask.to(int), dim=1).repeat(1, z_true.shape[1]).unsqueeze(-1)
        z_true_imputed  = torch.where(nan_mask, torch.tensor([2.0, 2.0], dtype=torch.float32, device=z_true.device) * escape_time, z_true)
        return z_true_imputed
    
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
        plt.imshow(magnitude, 
                extent=[real_min, real_max, imag_min, imag_max], 
                origin='lower', 
                cmap='hot',)

        plt.colorbar(label='Magnitude')
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
            c_complex = torch.view_as_complex(c)[0]
            z_real = [0]  # Ground truth values
            z_nn = [0]    # NN predicted values
            z = torch.tensor(0 + 0j, dtype=torch.complex64)

            # Generate ground truth and NN predictions
            for t in range(max_iter):
                z = z**2 + c_complex
                
                if torch.isnan(z.real) or torch.isnan(z.imag) or torch.abs(z) > 2:
                    break
                
                z_real.append(logscalemagnitude(z).cpu().numpy())
                t_tensor = torch.tensor([t], dtype=torch.float32, device=c.device).unsqueeze(0)  
                z_pred = self(t_tensor, c[[0]])[0]
                z_nn.append(logscalemagnitude(torch.view_as_complex(z_pred)).cpu().item())

            # Convert to numpy arrays for plotting
            z_real = np.array(z_real)
            z_pred = np.array(z_nn)

            # Create a color map for the time steps
            norm = mcolors.Normalize(vmin=0, vmax=t)
            cmap = cm.viridis
            colors = [cmap(norm(tt)) for tt in range(t + 1)]

            # Create a new figure with two subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

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
    dataset = MandelbrotDataset(samples=200_000, max_iter=100)
    val_dataset = MandelbrotDataset(samples=20_000, max_iter=100, fix_c=True)

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    model = MandelbrotNN(
        hidden_dim=2048,
        num_hidden_layers=6, 
        lr=1e-4, 
        max_iter=100,
        warmup_epochs=5, 
        total_epochs=1000,
        pde_weight=0.1,
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
        max_epochs=100_000,
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[lr_monitor, pl.callbacks.ModelSummary(max_depth=3), checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
