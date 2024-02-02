from time import perf_counter

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

SPINNER = "aesthetic"
REFRESH = 20
EXPAND = False

console = Console()


class ProgressBar:
    def __init__(self, indeterminate: bool = False) -> None:
        # Define custom progress bar
        if indeterminate:
            self.progress = Progress(
                # SpinnerColumn(spinner_name=SPINNER),
                TextColumn("🔄 [progress.description]{task.description}"),
                BarColumn(),
                transient=True,
                console=console,
                expand=EXPAND,
            )
        else:
            self.progress = Progress(
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                expand=EXPAND,
            )


class Timer:
    def __init__(self) -> None:
        self._start = None
        self._stop = None
        self.elapsed_s = None
        self.history = []

    def start(self) -> None:
        self._start = perf_counter()

    def stop(self) -> None:
        self._stop = perf_counter()

    def elapsed(self) -> None:
        self.elapsed_s = self._stop - self._start
        self.history.append(self.elapsed_s)
        console.print("⏳ Elapsed time: {:.2f} s".format(self.elapsed_s))
