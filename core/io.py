import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<mode>swv|cv)_ch(?P<ch>\d+)_([0-9a-f]+)_meas_"
    r"(?P<date>\d{8})_(?P<time>\d{4})_(?P<scan>\d+)_ch(?P<ch2>\d+)\.csv$",
    re.IGNORECASE,
)
CHANNEL_RE = re.compile(r"(?:^|[_\-\s])ch(?:annel)?\s*0*(\d+)(?:\D|$)", re.IGNORECASE)
SCAN_RE = re.compile(r"(?:^|[_\-\s])meas[_\-\s].*?(\d+)(?:\D|$)", re.IGNORECASE)
FILENAME_TIME_RE = re.compile(
    r"(?:^|_)(?P<date>\d{8})_(?P<time>\d{4}(?:\d{2})?)(?:_|\.|$)",
    re.IGNORECASE,
)


def parse_measurement_time_from_filename(filename: str) -> Optional[datetime]:
    """Extract a YYYYMMDD_HHMM[SS] measurement time from a filename."""
    match = FILENAME_TIME_RE.search(os.path.basename(filename))
    if match is None:
        return None
    try:
        time_format = "%Y%m%d%H%M%S" if len(match.group("time")) == 6 else "%Y%m%d%H%M"
        return datetime.strptime(
            f"{match.group('date')}{match.group('time')}", time_format
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class MeasurementFile:
    mode: str
    scan: int
    ch: int
    ts: int
    path: str
    folder_index: int
    measurement_time: Optional[datetime] = None


# Backward-compatible alias used across the existing SWV pipeline.
SWVFile = MeasurementFile


def collect_measurement_csvs_from_folders(
    folders: List[str],
    mode: Optional[str] = None,
) -> List[MeasurementFile]:
    wanted_mode = mode.lower() if mode else None
    out: List[MeasurementFile] = []
    for folder_index, folder in enumerate(folders):
        if os.path.isfile(folder):
            root = os.path.dirname(folder)
            filenames = [os.path.basename(folder)]
        elif os.path.isdir(folder):
            root = folder
            filenames = os.listdir(folder)
        else:
            raise ValueError(f"Input path not found: {folder}")
        for fn in filenames:
            if not fn.lower().endswith(".csv"):
                continue
            m = FILENAME_RE.match(fn)
            file_mode = m.group("mode").lower() if m else ("cv" if fn.lower().startswith("cv") else "swv")
            if wanted_mode is not None and file_mode != wanted_mode:
                continue
            if m:
                ch = int(m.group("ch"))
                if int(m.group("ch2")) != ch:
                    continue
                scan = int(m.group("scan"))
                ts = int(f"{m.group('date')}{m.group('time')}")
            else:
                channel_match = CHANNEL_RE.search(fn)
                scan_match = SCAN_RE.search(fn)
                ch = int(channel_match.group(1)) if channel_match else 1
                scan = int(scan_match.group(1)) if scan_match else 1
                ts = int(os.stat(os.path.join(root, fn)).st_mtime_ns)
            measurement_time = parse_measurement_time_from_filename(fn)
            out.append(
                MeasurementFile(
                    mode=file_mode,
                    scan=scan,
                    ch=ch,
                    ts=ts,
                    path=os.path.join(root, fn),
                    folder_index=folder_index,
                    measurement_time=measurement_time,
                )
            )
    return out


def collect_swv_csvs_from_folders(folders: List[str]) -> List[SWVFile]:
    return collect_measurement_csvs_from_folders(folders, mode="swv")


def collect_cv_csvs_from_folders(folders: List[str]) -> List[MeasurementFile]:
    return collect_measurement_csvs_from_folders(folders, mode="cv")


def group_by_channel_and_sort(files: List[SWVFile]) -> Dict[int, List[SWVFile]]:
    d: Dict[int, List[SWVFile]] = {}
    for f in files:
        d.setdefault(f.ch, []).append(f)
    for ch in d:
        d[ch].sort(key=lambda f: (f.folder_index, f.ts, f.scan))
    return d


def load_swv_csv(
    filepath: str,
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)

    if voltage_col not in df.columns:
        raise ValueError(
            f"Voltage column '{voltage_col}' not found. Columns: {list(df.columns)}"
        )

    if current_col is None:
        if "Current Diff (uA)" in df.columns:
            current_col = "Current Diff (uA)"
        elif "Current (uA)" in df.columns:
            current_col = "Current (uA)"
        else:
            raise ValueError(
                f"Cannot auto-pick current column. Need 'Current Diff (uA)' or 'Current (uA)'. "
                f"Columns: {list(df.columns)}"
            )
    elif current_col not in df.columns:
        raise ValueError(
            f"Current column '{current_col}' not found. Columns: {list(df.columns)}"
        )

    return df[voltage_col].to_numpy(dtype=float), df[current_col].to_numpy(dtype=float)


def filter_finite(v: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v, y = np.asarray(v, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(v) & np.isfinite(y)
    return v[mask], y[mask]
