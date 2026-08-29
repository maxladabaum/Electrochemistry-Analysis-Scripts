import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def _write_state(folder: Path, state: dict | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bo_state.json").write_text(
        json.dumps(state or {"session_id": folder.name, "observations": []}),
        encoding="utf-8",
    )
    return folder.resolve()


def test_discovery_finds_direct_sessions_without_entering_artifacts(tmp_path):
    sessions = tmp_path / "bo_sessions"
    first = _write_state(sessions / "first")
    second = _write_state(sessions / "second")
    # A state-looking file inside a simulated session artifact must not be
    # treated as another session.
    _write_state(first / "surrogate" / "large_artifact_tree" / "not_a_session")

    assert viewer.discover_bo_session_folders(tmp_path) == [first, second]


def test_discovery_prunes_artifacts_in_legacy_nested_layout(tmp_path):
    session = _write_state(tmp_path / "run" / "nested" / "legacy_session")
    _write_state(session / "simulation_artifacts" / "not_a_session")

    assert viewer.discover_bo_session_folders(tmp_path) == [session]


def test_large_session_label_does_not_parse_complete_state(tmp_path, monkeypatch):
    session = tmp_path / "bo_sessions" / "large_simulation"
    session.mkdir(parents=True)
    (session / "bo_state.json").write_bytes(b" " * (256 * 1024 + 1))

    def fail_if_read(_path):
        raise AssertionError("large state should not be parsed for a picker label")

    monkeypatch.setattr(viewer, "_read_json", fail_if_read)

    assert viewer._bo_session_choice_label(session, tmp_path) == (
        "bo_sessions/large_simulation"
    )
