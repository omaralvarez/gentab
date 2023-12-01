from . import Generator
from synthtab.console import console,SPINNER,REFRESH

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

class ROS(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.name = 'RandomOverSampler'

    def generate(self):
        with console.status(
            'Generating with {}...'.format(self.name), 
            spinner=SPINNER, 
            refresh_per_second=REFRESH
        ) as status:
            ros = RandomOverSampler(random_state=0)
            self.dataset.X_gen, self.dataset.y_gen = ros.fit_resample(self.dataset.X, self.dataset.y)
            #print(sorted(Counter(self.dataset.y).items()))
            #print(sorted(Counter(self.y_resampled).items()))
            #print(self.dataset.X.shape, self.dataset.y.shape)
            #print(X_resampled.shape, y_resampled.shape)
            #print(X_resampled, y_resampled)

        console.print('✅ Generation complete with {}...'.format(self.name))

    def __str__(self) -> str:
        return super().__str__() + ' Generator: ' + self.name
