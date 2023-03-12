GAN_MNIST = {
    "latent_dim": 64,
    "im_size": (28, 28),
    "batch_size": 128,
    "gen_hidden_dim": (128, 256, 512, 1024),
    "dis_hidden_dim": (512, 256, 128),
    "gen_lr": 2e-4,
    "gen_betas": (0.5, 0.999),
    "dis_lr": 2e-4,
    "dis_betas": (0.5, 0.999),
}
