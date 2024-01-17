from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

SPINNER = "aesthetic"
REFRESH = 20
EXPAND = False

console = Console()


class ProgressBar:
    def __init__(self, indeterminate=False) -> None:
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
