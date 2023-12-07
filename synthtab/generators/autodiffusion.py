from . import Generator
from .autodiff.process_GQ import DataFrameParser, convert_to_table
from .autodiff.autoencoder import train_autoencoder
from .autodiff.TabDDPMdiff import train_diffusion
from .autodiff.diffusion import Euler_Maruyama_sampling
from synthtab.console import console, SPINNER, REFRESH

import pandas as pd


# TODO TABDDPM & diffusion EulerMaruyama cuda in train_diffusion not working, check what is going on
# TODO Check that everything uses same device, right now it is a mess
# TODO (1) Stasy-AutoDiff : process_edited.py + diffusion.py + autoencoder.py
#      (2) Tab-AutoDiff : proccess_GQ.py + TabDDPMdiff.py + autoencoder.py


class AutoDiffusion(Generator):
    def __init__(
        self,
        dataset,
        threshold=0.01,
        device="cuda",
        max_tries_per_batch=4096,
        n_epochs=10,
        eps=1e-5,
        weight_decay=1e-6,
        maximum_learning_rate=1e-2,
        lr=2e-4,
        hidden_size=250,
        num_layers=3,
        batch_size=50,
        diff_n_epochs=1000,
        hidden_dims=(256, 512, 1024, 512, 256),
        sigma=20,
        num_batches_per_epoch=50,
        T=100,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "AutoDiffusion"
        self.threshold = threshold
        # Auto-encoder hyper-parameters
        self.device = device
        self.max_tries_per_batch = max_tries_per_batch
        self.n_epochs = n_epochs
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # diffusion hyper-parameters
        self.diff_n_epochs = diff_n_epochs
        self.hidden_dims = hidden_dims
        self.sigma = sigma
        self.num_batches_per_epoch = num_batches_per_epoch
        self.T = T
        self.data = self.dataset.get_single_df()

    def preprocess(self):
        # TODO right now it looks like this does nothing in the notebook, we should be sure
        self.parser = DataFrameParser().fit(self.data, self.threshold)

    def train(self) -> None:
        self.ds = train_autoencoder(
            self.data,
            self.hidden_size,
            self.num_layers,
            self.lr,
            self.weight_decay,
            self.n_epochs,
            self.batch_size,
            self.threshold,
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
        )

    def sample(self) -> pd.DataFrame:
        T = 300
        N = self.latent_features.shape[0]
        P = self.latent_features.shape[1]

        sample = Euler_Maruyama_sampling(self.score, T, N, P, self.device)
        gen_output = self.ds[0](sample, self.ds[2], self.ds[3])

        return convert_to_table(self.data, gen_output, self.threshold)

    def resample(self, n_samples) -> None:
        data_gen = self.data

        for _ in range(self.max_tries_per_batch):
            gen = self.sample()

            for cls, cnt in n_samples.items():
                if cnt > 0:
                    filtered = gen[gen[self.dataset.config["y_label"]] == cls]

                    count = len(filtered.index)
                    if count > cnt:
                        n_samples[cls] = 0
                        filtered = filtered.sample(n=cnt)
                    else:
                        n_samples[cls] = cnt - count

                    data_gen = pd.concat(
                        [data_gen, filtered], ignore_index=True, sort=False
                    )

            if sum(n_samples.values()) == 0:
                break

        self.dataset.set_split_result(data_gen)

    def balance(self) -> None:
        data_gen = self.data

        for _ in range(self.max_tries_per_batch):
            gen = self.sample()

            for cls, cnt in self.counts.items():
                if cnt > 0:
                    filtered = gen[gen[self.dataset.config["y_label"]] == cls]

                    count = len(filtered.index)
                    if count > cnt:
                        self.counts[cls] = 0
                        filtered = filtered.sample(n=cnt)
                    else:
                        self.counts[cls] = cnt - count

                    data_gen = pd.concat(
                        [data_gen, filtered], ignore_index=True, sort=False
                    )

            if self.counts.max() < 1:
                break

        self.dataset.set_split_result(data_gen)
