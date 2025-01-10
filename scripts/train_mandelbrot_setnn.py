# Main Training Loop
from mbsnn.dataset import MandelbrotDataset

from torch.utils.data import DataLoader

from mbsnn.mandelbrot_setnn import MandelbrotNN

import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig
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
    dataset = MandelbrotDataset(samples=cfg.dataset.num_train_samples, max_iter=cfg.dataset.max_iter, fix_c=False)
    val_dataset = MandelbrotDataset(samples=cfg.dataset.num_val_samples, max_iter=cfg.dataset.max_iter, fix_c=True)

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
        output_dir="predictions",)
    
    # add mlflow logger
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    wandb_logger = pl.loggers.WandbLogger(project="mandelbrot-set-nn", entity="andrinburli", log_model=True)
    
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
    #os.environ["WANDB_MODE"]="offline"
    
    hydra_training()