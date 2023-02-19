from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary
from torch.utils.data import DataLoader

from gpt.config import GPT_TINY
from gpt.data import MemoryMapDataset
from gpt.model import GPT


class LitGPT(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.model = GPT(**config).to(self.device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.model(tokens)
        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._calculate_loss(*batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._calculate_loss(*batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=2.5e-4)
        return optimizer

    def _calculate_loss(
        self, tokens: torch.Tensor, next_tokens: torch.Tensor
    ) -> torch.Tensor:
        tokens = tokens.to(self.device)
        next_tokens = next_tokens.to(self.device)
        next_tokens_pred = self.model(tokens)
        loss = F.cross_entropy(
            next_tokens_pred.view(-1, self.model.vocab_size), next_tokens.view(-1)
        )
        return loss


if __name__ == "__main__":
    variant = GPT_TINY

    train_dataset = MemoryMapDataset(
        Path(__file__).parent / "data" / "wikitext" / "train.bin", block_size=variant["max_context_size"]
    )
    val_dataset = MemoryMapDataset(
        Path(__file__).parent / "data" / "wikitext" / "val.bin", block_size=variant["max_context_size"]
    )

    model = LitGPT(variant)
    callbacks = [
        EarlyStopping(monitor="val_loss"),
        ModelCheckpoint(monitor="val_loss"),
        RichModelSummary(),
    ]
    logger = loggers.WandbLogger(project="gpt", log_model=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=256, pin_memory=True, num_workers=2),
        DataLoader(val_dataset, batch_size=64, pin_memory=True, num_workers=2),
    )
