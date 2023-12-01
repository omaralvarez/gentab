from . import Generator
from synthtab.console import console,SPINNER,REFRESH

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
import pandas as pd
from collections import Counter

class TVAE(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.name = 'TVAE'

    def train(self):
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

    def generate(self, num_rows):
        with console.status(
            'Training with {}...'.format(self.name), 
            spinner=SPINNER, 
            refresh_per_second=REFRESH
        ) as status:
            self.train()

            status.update(    
                'Generating with {}...'.format(self.name), 
                spinner=SPINNER
            )
            
            # TODO Check conditional sampling in docs to reduce imbalance
            data_gen = self.synthesizer.sample(num_rows)
            self.dataset.X_gen = data_gen.loc[:, data_gen.columns != self.dataset.config['y_label']]
            self.dataset.y_gen = data_gen[self.dataset.config['y_label']]

        console.print('✅ Generation complete with {}...'.format(self.name))

    def __str__(self) -> str:
        return super().__str__() + ' Generator: ' + self.name