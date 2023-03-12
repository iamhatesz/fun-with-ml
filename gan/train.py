from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from torchmetrics.classification import BinaryAccuracy
from torchvision.utils import make_grid

from gan.config import GAN_MNIST
from gan.data import MNISTDataModule
from gan.model import Discriminator, Generator


class GAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        im_size: tuple[int, ...],
        batch_size: int,
        gen_hidden_dim: tuple[int, ...],
        dis_hidden_dim: tuple[int, ...],
        gen_lr: float,
        gen_betas: tuple[float, float],
        dis_lr: float,
        dis_betas: tuple[float, float],
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self._im_size = im_size
        self._batch_size = batch_size
        self._gen_lr = gen_lr
        self._gen_betas = gen_betas
        self._dis_lr = dis_lr
        self._dis_betas = dis_betas

        self.generator = Generator(self._latent_dim, self._im_size, gen_hidden_dim).to(
            self.device
        )
        self.discriminator = Discriminator(self._im_size, dis_hidden_dim).to(
            self.device
        )

        self._last_real_batch: torch.Tensor | None = None

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        fake = self.generator(noise)
        return fake

    def configure_optimizers(self) -> list[optim.Optimizer]:
        gen_optimizer = optim.Adam(
            self.generator.parameters(), lr=self._gen_lr, betas=self._gen_betas
        )
        dis_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self._dis_lr, betas=self._dis_betas
        )
        return [gen_optimizer, dis_optimizer]

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor, optimizer_idx: int
    ) -> torch.Tensor:
        images, _ = batch
        images = images.to(self.device)
        # Generator
        if optimizer_idx == 0:
            loss = self._generator_loss()
            self.log("train_gen_loss", loss)
            return loss
        # Discriminator
        if optimizer_idx == 1:
            loss = self._discriminator_loss(images)
            self.log("train_dis_loss", loss)
            return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> None:
        reals, _ = batch
        reals = reals.to(self.device)
        reals_pred = self.discriminator(reals).detach()
        reals_labels = torch.ones_like(reals_pred)

        noise = self._noise(len(batch))
        fake = self.generator(noise).detach()
        fake_pred = self.discriminator(fake).detach()
        fake_labels = torch.zeros_like(fake_pred)

        metric = BinaryAccuracy().to(self.device)
        reals_accuracy = metric(reals_pred, reals_labels)
        fake_accuracy = metric(fake_pred, fake_labels)
        self.log("dis_reals_accuracy", reals_accuracy)
        self.log("dis_fake_accuracy", fake_accuracy)

    def _noise(self, num_samples: int) -> torch.Tensor:
        return torch.randn(num_samples, self._latent_dim, device=self.device)

    def _discriminator_loss(self, real: torch.Tensor) -> torch.Tensor:
        noise = self._noise(self._batch_size)
        fake = self.generator(noise).detach()
        fake_pred = self.discriminator(fake)
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        real_pred = self.discriminator(real)
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        loss = 0.5 * fake_loss + 0.5 * real_loss
        return loss

    def _generator_loss(self) -> torch.Tensor:
        noise = self._noise(self._batch_size)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake)
        loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self._last_real_batch = batch

    def on_train_epoch_end(self) -> None:
        # Log reals
        reals, _ = self._last_real_batch
        self.logger.log_image("reals", [wandb.Image(reals[:16])])

        # Log fakes
        noise = self._noise(16)
        fake = self.generator(noise).detach().cpu().view(-1, 1, *self._im_size)
        grid = make_grid(fake)
        self.logger.log_image("fakes", [wandb.Image(grid)])


if __name__ == "__main__":
    pl.seed_everything(2137, workers=True)

    config = GAN_MNIST
    model = GAN(**config)
    dm = MNISTDataModule(
        Path(__file__).parent / "data", batch_size=config["batch_size"], num_workers=4
    )
    callbacks = [
        RichModelSummary(),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = loggers.WandbLogger(project="gan", log_model=True)
    logger.watch(model)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, dm)
