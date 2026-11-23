import csv
from datetime import datetime, timezone

from app.config.settings import settings
from app.utils.logger import logger

FIELDS = ["timestamp", "query", "model", "mode", "latency"]


class MetricsLogger:
    """Appends one row per query to reports/metrics.csv (best effort)."""

    def __init__(self):
        self.file_path = settings.METRICS_FILE

    def log(self, query, model, latency, mode="llm"):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.file_path.exists()
            with open(self.file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(FIELDS)
                writer.writerow(
                    [datetime.now(timezone.utc).isoformat(timespec="seconds"), query, model or "", mode, latency]
                )
        except OSError as exc:  # read-only filesystems must not break answering
            logger.warning("Could not write metrics: %s", exc)
