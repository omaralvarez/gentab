from rich import print

class Generator:
    def __init__(self, algo, dataset) -> None:
        self.algo = algo
        self.dataset = dataset

        print("✅ Creating generator with {}...".format(self.algo))

    def __str__(self) -> str:
        return str(self.algo)