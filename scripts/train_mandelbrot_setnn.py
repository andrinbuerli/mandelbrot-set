# Main Training Loop
from mbsnn.dataset import MandelbrotDataset

from torch.utils.data import DataLoader

from mbsnn.mandelbrot_setnn import MandelbrotNN

import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
    
@hydra.main(
    config_path=str(
        Path(__file__).parent.parent.resolve()
        / "resources" / "configs"
    ),
    config_name="",
    version_base="1.1",
)
def hydra_training(cfg: DictConfig = None) -> None:
    dataset = MandelbrotDataset(
        samples=cfg.dataset.num_train_samples,
        max_iter=cfg.dataset.max_iter,
        fix_c=False,
        max_magnitude=cfg.dataset.max_magnitude,
        biased_sampling=cfg.dataset.biased_sampling,
        real_range=cfg.dataset.real_range,
        imag_range=cfg.dataset.imag_range,
        log_scale=cfg.dataset.log_scale)
    val_dataset = MandelbrotDataset(
        samples=cfg.dataset.num_val_samples,
        max_iter=cfg.dataset.max_iter,
        fix_c=True,
        max_magnitude=cfg.dataset.max_magnitude,
        biased_sampling=cfg.dataset.biased_sampling,
        real_range=cfg.dataset.real_range,
        imag_range=cfg.dataset.imag_range,
        log_scale=cfg.dataset.log_scale)

    train_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.dataset.num_train_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.dataset.num_val_workers, persistent_workers=True)

    model = MandelbrotNN(
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers, 
        lr=cfg.train.lr, 
        max_iter=cfg.dataset.max_iter,
        warmup_epochs=cfg.train.warmup_epochs, 
        total_epochs=cfg.train.total_epochs,
        pde_weight=cfg.train.pde_weight,
        max_batch_size=cfg.train.max_batch_size,
        max_magnitude=cfg.dataset.max_magnitude,
        time_token_dim=cfg.model.get("time_token_dim", None),
        compute_dz_dc=cfg.model.compute_dz_dc,
        compute_dz_dt=cfg.model.compute_dz_dt,
        log_scale=cfg.dataset.log_scale,
        pde_weight_warmup_epochs=cfg.train.pde_weight_warmup_epochs,
        use_clipped_nans=cfg.train.use_clipped_nans,
        output_dir="predictions",)
    
    # add mlflow logger
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    wandb_logger = pl.loggers.WandbLogger(project="mandelbrot-set-nn", entity="andrinburli", log_model=True)
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))
    #wandb_logger.watch(model, log="all", log_freq=1000)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints',
        filename='mandelbrot-{epoch:02d}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.total_epochs,
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[lr_monitor, pl.callbacks.ModelSummary(max_depth=3), checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    

if __name__ == "__main__":
    import os
    
    if os.environ.get("WANDB_API_KEY") is None:
        os.environ["WANDB_MODE"]="offline"
    
    hydra_training()