from . import Generator
from synthtab.console import console,SPINNER,REFRESH

from imblearn.over_sampling import SMOTE as sm
from collections import Counter

class SMOTE(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.name = 'SMOTE'

    def generate(self):
        with console.status(
            'Generating with {}...'.format(self.name), 
            spinner=SPINNER, 
            refresh_per_second=REFRESH
        ) as status:
            self.dataset.X_gen, self.dataset.y_gen = sm().fit_resample(self.dataset.X, self.dataset.y)

        #console.print(sorted(Counter(self.dataset.y).items()))
        #console.print(sorted(Counter(self.dataset.y_gen).items()))

        console.print('✅ Generation complete with {}...'.format(self.name))

    def __str__(self) -> str:
        return super().__str__() + ' Generator: ' + self.name