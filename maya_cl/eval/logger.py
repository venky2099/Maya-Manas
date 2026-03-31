# logger.py -- Maya-Manas (Paper 7)
# Extends Paper 6 logger with manas_peak_fraction column.

import csv
import os
from datetime import datetime
from maya_cl.utils.config import RESULTS_DIR


class RunLogger:
    def __init__(self, run_name: str):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"{run_name}_{ts}_batches.csv")
        self._file   = open(path, "w", newline="")
        self._writer = None
        self._path   = path
        print(f"Logger: {path}")

    def log_batch(self, task: int, epoch: int, batch: int,
                  loss: float, confidence: float, pain_fired: bool,
                  lability_mean: float, vairagya_protection: float,
                  affective: dict,
                  samskara_mean: float = 0.0,
                  moha_fraction: float = 0.0,
                  retrograde_fired: bool = False,
                  manas_peak_fraction: float = 0.0) -> None:
        row = {
            "task":                    task,
            "epoch":                   epoch,
            "batch":                   batch,
            "loss":                    round(loss, 6),
            "confidence":              round(confidence, 6),
            "pain_fired":              int(pain_fired),
            "lability_mean":           round(lability_mean, 6),
            "vairagya_protection_fc1": round(vairagya_protection, 6),
            "shraddha":                round(affective.get("shraddha", 0), 6),
            "bhaya":                   round(affective.get("bhaya",    0), 6),
            "vairagya":                round(affective.get("vairagya", 0), 6),
            "spanda":                  round(affective.get("spanda",   0), 6),
            "viveka_signal":           round(affective.get("viveka",   0), 6),
            "buddhi":                  round(affective.get("buddhi",   0), 6),
            "chitta":                  round(affective.get("chitta",   0), 6),
            "manas":                   round(affective.get("manas",    0), 6),
            "samskara_mean":           round(samskara_mean,        6),
            "moha_fraction":           round(moha_fraction,        6),
            "retrograde_fired":        int(retrograde_fired),
            "manas_peak_fraction":     round(manas_peak_fraction,  6),
        }
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)

    def log_task_summary(self, task_id: int,
                         acc_dict: dict, summary: dict) -> None:
        print(f"  [Task {task_id} summary] "
              f"AA={summary.get('AA','?')} "
              f"BWT={summary.get('BWT','?')}")

    def log_final(self, summary: dict) -> None:
        print(f"\n  FINAL -- AA:{summary.get('AA')} "
              f"BWT:{summary.get('BWT')} "
              f"FWT:{summary.get('FWT')}")

    def close(self) -> None:
        self._file.close()
