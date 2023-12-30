from synthtab import SEED
from synthtab.utils import console, SPINNER, REFRESH, ProgressBar

import pandas as pd


class Generator:
    def __init__(self, dataset, batch_size=1000, max_tries_per_batch=1000) -> None:
        self.batch_size = batch_size
        self.max_tries_per_batch = max_tries_per_batch
        # Reset generated dataframes
        self.dataset = dataset
        self.config = dataset.config
        self.dataset.X_gen = None
        self.dataset.y_gen = None
        # Get class counts
        self.orig_counts = dataset.class_counts()
        # Get max, subtract it to the rest to get target samples
        self.counts = self.orig_counts.max() - self.orig_counts
        # Filter the maximum class no need to generate for it
        self.counts = self.counts[self.counts > 0]
        # For reproducibility
        self.seed = SEED

    def __str__(self) -> str:
        return self.__class__.__name__

    def preprocess(self) -> None:
        pass

    def train(self) -> None:
        pass

    def sample(self) -> pd.DataFrame:
        pass

    def save_to_disk(self) -> None:
        self.dataset.save_to_disk(self)

    def load_from_disk(self) -> None:
        self.dataset.load_from_disk(self)

    def resample(self, n_samples) -> None:
        data_gen = self.dataset.get_single_df()

        total_samples = sum(n_samples.values())

        with ProgressBar().progress as p:
            gen_task = p.add_task(
                "Generating with {}...".format(self), total=total_samples
            )
            for i in range(self.max_tries_per_batch):
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

                missing_samples = sum(n_samples.values())

                p.update(gen_task, completed=total_samples - missing_samples)

                if missing_samples == 0:
                    break
                elif i == self.max_tries_per_batch - 1:
                    raise RuntimeError(
                        "Maximum number of tries reached, model probably did not"
                        " converge."
                    )

        self.dataset.set_split_result(data_gen)

    def balance(self) -> None:
        data_gen = self.dataset.get_single_df()

        total_samples = self.counts.sum()

        with ProgressBar().progress as p:
            gen_task = p.add_task(
                "Generating with {}...".format(self), total=total_samples
            )

            for i in range(self.max_tries_per_batch):
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

                missing_samples = self.counts.sum()

                p.update(gen_task, completed=total_samples - missing_samples)

                if self.counts.max() < 1:
                    break
                elif i == self.max_tries_per_batch - 1:
                    raise RuntimeError(
                        "Maximum number of tries reached, model probably did not"
                        " converge."
                    )

        self.dataset.set_split_result(data_gen)

    # TODO Flag to append or just leave generated
    def generate(self, n_samples=None) -> None:
        with console.status(
            "Preprocessing {}...".format(self.dataset),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            self.preprocess()

            status.update("Training with {}...".format(self), spinner=SPINNER)
            self.train()

            # status.update(
            #     "Generating with {}...".format(self), spinner=SPINNER
            # )

        console.print("🔄 Generating with {}...".format(self))

        # If progress does not look better indent this
        if n_samples is None:
            self.balance()
        else:
            self.resample(n_samples)

        console.print("✅ Generation complete with {}...".format(self))
