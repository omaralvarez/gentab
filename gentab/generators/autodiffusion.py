from . import Generator
from .autodiff.process_GQ import DataFrameParser, convert_to_table
from .autodiff.autoencoder import train_autoencoder
from .autodiff.TabDDPMdiff import train_diffusion
from .autodiff.diffusion import Euler_Maruyama_sampling
from gentab.utils import console, PROG_COLUMNS, DEVICE

import pandas as pd


class AutoDiffusion(Generator):
    def __init__(
        self,
        dataset,
        threshold=0.01,
        device=DEVICE,
        max_tries_per_batch=8192,
        n_epochs=10000,
        eps=1e-5,
        weight_decay=1e-6,
        maximum_learning_rate=1e-2,
        lr=2e-4,
        hidden_size=250,
        num_layers=3,
        batch_size=100,
        diff_n_epochs=10000,
        hidden_dims=(256, 512, 1024, 512, 256),
        sigma=20,
        num_batches_per_epoch=50,
        T=100,
    ) -> None:
        super().__init__(dataset, batch_size, max_tries_per_batch)
        self.threshold = threshold
        # Auto-encoder hyper-parameters
        self.device = device
        self.n_epochs = n_epochs
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # diffusion hyper-parameters
        self.diff_n_epochs = diff_n_epochs
        self.hidden_dims = hidden_dims
        self.sigma = sigma
        self.num_batches_per_epoch = num_batches_per_epoch
        self.T = T
        self.data = self.dataset.get_single_df()

    def preprocess(self) -> None:
        super().preprocess()

    def train(self) -> None:
        # Setup progress
        self.p.columns = PROG_COLUMNS
        self.p.update(
            self.gen_task,
            total=self.n_epochs,
            description="Training {} AE...".format(self),
        )
        dm_task = self.p.add_task(
            total=self.diff_n_epochs,
            description="Training {} DM...".format(self),
        )

        self.ds = train_autoencoder(
            self.data,
            self.hidden_size,
            self.num_layers,
            self.lr,
            self.weight_decay,
            self.n_epochs,
            self.batch_size,
            self.threshold,
            self.p,
            self.gen_task,
        )
        self.latent_features = self.ds[1].detach()
        self.converted_table_dim = self.latent_features.shape[1]

        self.score = train_diffusion(
            self.latent_features,
            self.T,
            self.eps,
            self.sigma,
            self.lr,
            self.num_batches_per_epoch,
            self.maximum_learning_rate,
            self.weight_decay,
            self.diff_n_epochs,
            self.batch_size,
            self.device,
            self.p,
            dm_task,
        )

    def sample(self) -> pd.DataFrame:
        T = self.T
        N = self.latent_features.shape[0]
        P = self.latent_features.shape[1]

        sample = Euler_Maruyama_sampling(self.score, T, N, P, self.device)
        gen_output = self.ds[0](sample, self.ds[2], self.ds[3])

        return convert_to_table(self.data, gen_output, self.threshold)
