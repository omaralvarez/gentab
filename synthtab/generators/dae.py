from . import Generator
from synthtab.utils import console, SPINNER, REFRESH

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition
import pandas as pd
from collections import Counter


class DAE(Generator):
    def __init__(
        self,
        dataset,
        body_network="deepstack",
        body_network_cfg=dict(hidden_size=128),
        swap_noise_probas=0.2,
        cats_handling="onehot",
        cards=[],
        embeded_dims=[],
        device="cuda",
        batch_size=8192,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "DAE"
        self.body_network = body_network
        self.body_network_cfg = body_network_cfg
        self.swap_noise_probas = swap_noise_probas
        self.cats_handling = cats_handling
        self.cards = cards
        self.embeded_dims = embeded_dims
        self.device = device
        self.batch_size = batch_size
        self.max_tries_per_batch = max_tries_per_batch

        self.dae = DAE(
            body_network=self.body_network,
            body_network_cfg=self.body_network_cfg,
            swap_noise_probas=self.swap_noise_probas,
            cards=self.cards,
            cats_handling=self.cats_handling,
            embeded_dims=self.embeded_dims,
            device=self.device,
        )

    def preprocess(self) -> None:
        self.data = self.dataset.get_single_df()

    def train(self) -> None:
        self.dae.fit(
            self.data,
            verbose=1,
            batch_size=self.batch_size,
            optimizer_params={"lr": 3e-4},
        )

    def sample(self) -> pd.DataFrame:
        # sample = self.synthesizer.sample(self.batch_size)
        return None
