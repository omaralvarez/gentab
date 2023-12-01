from rich.console import Console

class Generator:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.console = Console()

    def __str__(self) -> str:
        return str(self.dataset.config['name'])