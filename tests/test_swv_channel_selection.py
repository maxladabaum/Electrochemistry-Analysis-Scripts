"""Channel selection must limit processing, not merely returned plots."""
from datetime import datetime, timedelta

import pytest

from core import analysis
from core.io import SWVFile


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("selection", [None, [2]])
@pytest.mark.parametrize("use_time_range", [False, True])
def test_channel_filter_limits_work_and_progress(monkeypatch, workers, selection, use_time_range):
    start = datetime(2026, 9, 25, 12)
    files = [
        SWVFile(mode="swv", scan=scan, ch=channel, ts=scan,
                path=f"/input/ch{channel}_{scan}.csv", folder_index=0,
                measurement_time=start + timedelta(minutes=scan))
        for channel in [1, 2, 3] for scan in [1, 2]
    ]
    monkeypatch.setattr(analysis, "collect_swv_csvs_from_folders", lambda folders: files)
    indexed = []
    monkeypatch.setattr(analysis, "_build_method_file_index", lambda selected: indexed.extend(selected) or {})
    monkeypatch.setattr(analysis, "_infer_method_path", lambda *args: None)
    monkeypatch.setattr(analysis, "_infer_method_path_direct", lambda *args: None)
    processed = []

    def process(**kwargs):
        processed.append(kwargs["measurement"])
        return {}, {"status": "OK", "result": {
            "status": "OK", "peak_voltage": -0.2, "bracket_width_V": 0.1,
            "skew": 0.0, "peak_offset_norm": 0.5,
        }}

    monkeypatch.setattr(analysis, "_process_swv_work_item", process)
    progress = []
    rows = analysis.run_batch(
        ["/input"], channels=selection, parallel_workers=workers,
        time_range=(start, start + timedelta(minutes=3)) if use_time_range else None,
        progress_callback=lambda done, total, name: progress.append((done, total)),
    )
    expected = {2} if selection else {1, 2, 3}
    assert {file.ch for file in processed} == expected
    assert {row["channel"] for row in rows} == expected
    assert len(rows) == len(expected) * 2
    assert progress[-1] == (len(rows), len(rows))
    if not use_time_range:
        assert {file.ch for file in indexed} == expected
        assert [row["scan_number"] for row in rows if row["channel"] == 2] == [1, 2]


def test_unmatched_channel_does_not_process_any_files(monkeypatch):
    files = [SWVFile(mode="swv", scan=1, ch=1, ts=1, path="/input/ch1.csv", folder_index=0)]
    monkeypatch.setattr(analysis, "collect_swv_csvs_from_folders", lambda folders: files)
    monkeypatch.setattr(analysis, "_process_swv_work_item", lambda **kwargs: pytest.fail("Unexpected processing"))
    with pytest.raises(ValueError, match="selected channels"):
        analysis.run_batch(["/input"], channels=[99])
