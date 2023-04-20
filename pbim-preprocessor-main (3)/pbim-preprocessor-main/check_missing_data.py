import argparse
import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start", type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"), required=True
)
parser.add_argument(
    "--end", type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"), required=True
)
parser.add_argument("--path", type=Path, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    current = args.start
    while current <= args.end:
        path = (
            args.path / f"{current.week}" / f"{current.month:02}" / f"{current.day:02}"
        )
        current_start = current
        while path.exists():
            current += datetime.timedelta(days=1)
            path = (
                args.path
                / f"{current.week}"
                / f"{current.month:02}"
                / f"{current.day:02}"
            )
        if current - current_start > datetime.timedelta(days=1):
            print(
                f"{current_start.date()} - {(current - datetime.timedelta(days=1)).date()}: {(current - current_start).days} days"
            )
        current += datetime.timedelta(days=1)
