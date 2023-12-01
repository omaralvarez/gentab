from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

class ProgressBar:
    def __init__(self) -> None:
        # Define custom progress bar
        self.progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

# |usage sample|
# with ProgresBar().progress as p:
#     # for i in track(range(20), description="Processing..."):
#     #     time.sleep(1)  # Simulate work being done
#     for n in p.track(range(1000)):
#         n = n - 2
#         total = n + total

# print(total)