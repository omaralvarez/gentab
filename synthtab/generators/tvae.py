from . import Generator
from synthtab.console import console,SPINNER,REFRESH

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition
import pandas as pd
from collections import Counter

class TVAE(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.__name__ = 'TVAE'

    def train(self) -> None:
        data = pd.concat([self.dataset.X, self.dataset.y], axis=1)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        # console.print(metadata.to_dict())
        # TODO for more options expand VAE additional custom options
        self.synthesizer = TVAESynthesizer(
            metadata, # required
            enforce_min_max_values=True,
            enforce_rounding=False,
            epochs=100,
            batch_size=4096,
            compress_dims=(128, 128),
            decompress_dims=(128, 128),
            embedding_dim=128,
            cuda=True
        )
        self.synthesizer.fit(data)

    def sample(self) -> None:
        conditions = []
        for cls, cnt in self.counts.items():
            conditions.append(Condition(
                num_rows=cnt,
                column_values = {self.dataset.config['y_label']: cls}
            ))

        data_gen = self.synthesizer.sample_from_conditions(
            conditions=conditions,
            batch_size=4096,
            max_tries_per_batch=4096
        )
        
        self.dataset.X_gen = data_gen.loc[:, data_gen.columns != self.dataset.config['y_label']]
        self.dataset.y_gen = data_gen[self.dataset.config['y_label']]
