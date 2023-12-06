from synthtab.console import console, SPINNER, REFRESH


class Generator:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        # Get class counts
        self.counts = dataset.class_counts()
        # Get max, subtract it to the rest to get target samples
        self.counts = self.counts.max() - self.counts
        # Filter the maximum class no need to generate for it
        self.counts = self.counts[self.counts > 0]

    def generate(self) -> None:
        with console.status(
            "Training with {}...".format(self.__name__),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            self.train()

            status.update(
                "Generating with {}...".format(self.__name__), spinner=SPINNER
            )

            self.sample()

        console.print("✅ Generation complete with {}...".format(self.__name__))

    def __str__(self) -> str:
        return self.__name__
