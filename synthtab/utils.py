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
import humanize
import datetime as dt

SPINNER = "aesthetic"
REFRESH = 20
EXPAND = False
TRANSIENT = True
IND_COLUMNS = [  # SpinnerColumn(spinner_name=SPINNER),
    TextColumn("🔄 [progress.description]{task.description}"),
    BarColumn(),
]

PROG_COLUMNS = [
    TextColumn("🔄 [progress.description]{task.description}"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
]

console = Console()


class ProgressBar:
    def __init__(self, indeterminate: bool = False) -> None:
        # Define custom progress bar
        if indeterminate:
            self.progress = Progress(
                *IND_COLUMNS,
                transient=TRANSIENT,
                console=console,
                expand=EXPAND,
            )
        else:
            self.progress = Progress(
                *PROG_COLUMNS,
                transient=TRANSIENT,
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
        console.print(
            "⏳ Elapsed time: {}".format(
                humanize.precisedelta(dt.timedelta(seconds=self.elapsed_s))
            )
        )

    def clear_history(self):
        self.history.clear()
