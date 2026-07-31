"""Convert MATLAB voltage/current matrices into the app's native SWV CSV layout."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


MAT_NAME_RE = re.compile(
    r"ch[_-]?(?P<channel>\d+).*?scan[_-]?(?P<scan>\d+).*?"
    r"(?P<frequency>\d+(?:\.\d+)?)hz",
    re.IGNORECASE,
)


def _load_voltage_current(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    contents = loadmat(path)
    candidates = []
    for name, value in contents.items():
        if name.startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
            candidates.append((name, np.asarray(array, dtype=float)))
    if not candidates:
        raise ValueError("no numeric 2D matrix found")

    variable, matrix = next(
        (candidate for candidate in candidates if candidate[0] == "data"),
        candidates[0],
    )
    if matrix.shape[0] == 2:
        voltage, current = matrix[0], matrix[1]
    elif matrix.shape[1] == 2:
        voltage, current = matrix[:, 0], matrix[:, 1]
    else:
        raise ValueError(
            f"variable '{variable}' has shape {matrix.shape}; expected 2×N or N×2"
        )

    finite = np.isfinite(voltage) & np.isfinite(current)
    voltage = np.asarray(voltage[finite], dtype=float)
    current = np.asarray(current[finite], dtype=float)
    if voltage.size < 5:
        raise ValueError("fewer than five finite voltage/current points")
    return variable, voltage, current


def _method_text(voltage: np.ndarray, frequency_hz: float) -> str:
    start_mv = float(voltage[0] * 1000.0)
    end_mv = float(voltage[-1] * 1000.0)
    steps = np.abs(np.diff(voltage))
    step_mv = float(np.median(steps[steps > 0]) * 1000.0) if np.any(steps > 0) else 0.0
    # This line follows the native method parser's meas_loop_swv format. All five
    # settings are retained for provenance and settings-based grouping.
    return (
        "meas_loop_swv imported mat file data "
        f"{start_mv:g}m {end_mv:g}m {step_mv:g}m 0m {frequency_hz:g}\n"
    )


def convert_mat_folders_to_swv_csv(
    folders: Iterable[str],
    output_name: str = "_mat_csv",
) -> dict:
    """Convert recursively discovered MAT files and return a conversion report."""
    output_folders: list[str] = []
    converted: list[dict] = []
    failed: list[dict] = []

    for folder_text in folders:
        source_root = Path(folder_text).expanduser().resolve()
        if not source_root.is_dir():
            failed.append({"file": str(source_root), "error": "folder not found"})
            continue

        mat_files = sorted(
            (
                path
                for path in source_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() == ".mat"
                and output_name not in path.parts
            ),
            key=lambda path: str(path.relative_to(source_root)).lower(),
        )
        if not mat_files:
            continue

        output_folder = source_root / output_name
        methods_folder = output_folder / "methods_used"
        output_folder.mkdir(parents=True, exist_ok=True)
        methods_folder.mkdir(parents=True, exist_ok=True)
        output_folders.append(str(output_folder))

        for mat_path in mat_files:
            relative = mat_path.relative_to(source_root)
            match = MAT_NAME_RE.search(mat_path.stem)
            if not match:
                failed.append(
                    {
                        "file": str(mat_path),
                        "error": "filename does not contain channel, scan, and frequency",
                    }
                )
                continue
            try:
                _variable, voltage, current = _load_voltage_current(mat_path)
                channel = int(match.group("channel"))
                scan = int(match.group("scan"))
                frequency_hz = float(match.group("frequency"))
                digest = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:12]
                stem = (
                    f"swv_ch{channel}_{digest}_meas_20000101_0000_"
                    f"{scan}_ch{channel}"
                )
                csv_path = output_folder / f"{stem}.csv"
                method_path = methods_folder / f"{stem}.ms"

                pd.DataFrame(
                    {
                        "Potential (V)": voltage,
                        "Current Diff (uA)": current,
                    }
                ).to_csv(csv_path, index=False)
                method_path.write_text(
                    _method_text(voltage, frequency_hz),
                    encoding="utf-8",
                )
                converted.append(
                    {
                        "source": str(mat_path),
                        "csv": str(csv_path),
                        "channel": channel,
                        "scan": scan,
                        "frequency_hz": frequency_hz,
                    }
                )
            except Exception as exc:
                failed.append({"file": str(mat_path), "error": str(exc)})

    return {
        "output_folders": output_folders,
        "converted": converted,
        "failed": failed,
    }
