from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import RichModelSummary
from torchmetrics.classification import BinaryAccuracy
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from dcgan.model import Discriminator, Generator, weights_init
from gan.data import MNISTDataModule
from wgan_gp.config import WGAN_GP_MNIST


class WGAN_GP(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        batch_size: int,
        gen_hidden_dim: int,
        dis_hidden_dim: int,
        gen_lr: float,
        gen_betas: tuple[float, float],
        dis_lr: float,
        dis_betas: tuple[float, float],
        lambda_: float,
        dis_steps: int,
    ):
        super().__init__()
        self.automatic_optimization = False

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._gen_lr = gen_lr
        self._gen_betas = gen_betas
        self._dis_lr = dis_lr
        self._dis_betas = dis_betas
        self._lambda = lambda_
        self._dis_steps = dis_steps

        self.generator = Generator(self._latent_dim, gen_hidden_dim, num_channels).to(
            self.device
        )
        self.discriminator = Discriminator(dis_hidden_dim, num_channels).to(self.device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

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
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        gen_optimizer, dis_optimizer = self.optimizers()

        for _ in range(self._dis_steps):
            dis_optimizer.zero_grad()
            images, _ = batch
            images = images.to(self.device)
            dis_loss = self._discriminator_loss(images)
            self.log("train_dis_loss", dis_loss)
            self.manual_backward(dis_loss)
            dis_optimizer.step()

        gen_optimizer.zero_grad()
        gen_loss = self._generator_loss()
        self.log("train_gen_loss", gen_loss)
        self.manual_backward(gen_loss)
        gen_optimizer.step()

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
        return torch.randn(num_samples, self._latent_dim, 1, 1, device=self.device)

    def _discriminator_loss(self, real: torch.Tensor) -> torch.Tensor:
        noise = self._noise(len(real))
        fake = self.generator(noise).detach()
        fake_pred = self.discriminator(fake)
        fake_loss = torch.mean(fake_pred)
        self.log("train_fake_loss", fake_loss)

        real_pred = self.discriminator(real)
        real_loss = -torch.mean(real_pred)
        self.log("train_real_loss", real_loss)

        grad_penalty = self._gradient_penalty(real, fake)
        self.log("train_grad_penalty", grad_penalty)

        loss = fake_loss + real_loss + self._lambda * grad_penalty
        return loss

    def _generator_loss(self) -> torch.Tensor:
        noise = self._noise(self._batch_size)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake)
        loss = -torch.mean(fake_pred)
        return loss

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        grad = self._gradient(real, fake.detach())
        grad = grad.view(len(grad), -1)
        grad_norm = grad.norm(2, dim=1)
        penalty = torch.mean(torch.pow(grad_norm - 1, 2))
        return penalty

    def _gradient(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        epsilon = torch.rand(len(real), 1, 1, 1, device=self.device, requires_grad=True)
        mixed = real * epsilon + fake * (1 - epsilon)
        score = self.discriminator(mixed)
        grad = torch.autograd.grad(
            inputs=mixed,
            outputs=score,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self._last_real_batch = batch

    def on_train_epoch_end(self) -> None:
        # Log reals
        reals, _ = self._last_real_batch
        self.logger.log_image("reals", [wandb.Image(reals[:16])])

        # Log fakes
        noise = self._noise(16)
        fake = self.generator(noise).detach().cpu()
        grid = make_grid(fake, normalize=True, value_range=(-1, 1))
        self.logger.log_image("fakes", [wandb.Image(grid)])


if __name__ == "__main__":
    pl.seed_everything(2137, workers=True)

    config = WGAN_GP_MNIST
    model = WGAN_GP(**config)
    dm = MNISTDataModule(
        Path(__file__).parent / "data",
        batch_size=config["batch_size"],
        num_workers=4,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    callbacks = [
        RichModelSummary(),
    ]
    logger = loggers.WandbLogger(project="wgan-gp", log_model=True)
    logger.watch(model)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, dm)
