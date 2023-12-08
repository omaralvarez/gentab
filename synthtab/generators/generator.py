from synthtab.console import console, SPINNER, REFRESH
from synthtab import SEED

import pandas as pd


class Generator:
    def __init__(self, dataset, batch_size=1000, max_tries_per_batch=1000) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tries_per_batch = max_tries_per_batch
        # Get class counts
        self.orig_counts = dataset.class_counts()
        # Get max, subtract it to the rest to get target samples
        self.counts = self.orig_counts.max() - self.orig_counts
        # Filter the maximum class no need to generate for it
        self.counts = self.counts[self.counts > 0]
        # For reproducibility
        self.seed = SEED

    def __str__(self) -> str:
        return self.__name__

    def preprocess(self) -> None:
        pass

    def train(self) -> None:
        pass

    def sample(self) -> pd.DataFrame:
        pass

    def resample(self, n_samples) -> None:
        data_gen = self.dataset.get_single_df()

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
        data_gen = self.dataset.get_single_df()

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

    def generate(self, n_samples=None) -> None:
        with console.status(
            "Preprocessing {}...".format(self.dataset),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            self.preprocess()

            status.update("Training with {}...".format(self.__name__), spinner=SPINNER)
            self.train()

            status.update(
                "Generating with {}...".format(self.__name__), spinner=SPINNER
            )
            if n_samples is None:
                self.balance()
            else:
                self.resample(n_samples)

        console.print("✅ Generation complete with {}...".format(self.__name__))
