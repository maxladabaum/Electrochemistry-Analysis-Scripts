import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import bo_session_viewer as viewer


@pytest.mark.parametrize('enabled', [False, True])
def test_simulated_sweep_sections_require_explicit_loading(monkeypatch, tmp_path, enabled):
    session = {
        'root': tmp_path,
        'state': {'simulation_metadata': {'run_count': 20}},
        'config': {'records': {'simulated_session': True}},
    }
    calls = []
    def checkbox(label, **kwargs):
        calls.append((label, kwargs))
        return enabled
    monkeypatch.setattr(viewer.st, 'checkbox', checkbox)
    assert viewer._load_simulated_sweep_section(session, 'Surrogate plots') is enabled
    assert calls[0][1]['value'] is False
    assert 'Surrogate plots' in calls[0][0]


def test_real_sessions_keep_existing_loading_behavior(monkeypatch, tmp_path):
    monkeypatch.setattr(viewer.st, 'checkbox', lambda *args, **kwargs: pytest.fail('Unexpected gate'))
    assert viewer._load_simulated_sweep_section({'root': tmp_path}, 'Surrogate plots')


@pytest.mark.parametrize('function', [
    '_render_swv_traces_tab', '_render_real_data_landscapes_tab',
    '_render_surrogate_tab', '_render_simulation_tab',
])
def test_disabled_fragments_return_before_loading_or_plotting(function):
    tree = ast.parse(Path(viewer.__file__).read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function)
    node.decorator_list = []
    namespace = {
        'full_session': {},
        '_load_simulated_sweep_section': lambda *args: False,
        'st': SimpleNamespace(session_state={}),
    }
    # No data/plot helpers are supplied: reaching any of that work would fail.
    exec(compile(ast.Module(body=[node], type_ignores=[]), viewer.__file__, 'exec'), namespace)
    assert namespace[function]() is None


def test_surrogate_groups_follow_observation_scope_without_stale_selections(monkeypatch, tmp_path):
    tree = ast.parse(Path(viewer.__file__).read_text())
    node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == '_render_surrogate_tab'
    )
    node.decorator_list = []
    stop = next(
        index for index, statement in enumerate(node.body)
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == 'not selected_surrogates'
    )
    node.body = node.body[:stop] + [
        ast.Return(value=ast.Name(id='selected_surrogates', ctx=ast.Load()))
    ]
    ast.fix_missing_locations(node)
    state = {}
    calls = []
    def multiselect(label, options, **kwargs):
        calls.append((options, [kwargs['format_func'](value) for value in options]))
        return state[kwargs['key']]
    fake_st = SimpleNamespace(session_state=state, multiselect=multiselect)
    monkeypatch.setattr(viewer, 'st', fake_st)
    groups = [
        {'id': 1, 'name': 'Ch 9 | initial=0', 'channels': [9]},
        {'id': 2, 'name': 'Ch 9 | initial=0', 'channels': [9]},
        {'id': 3, 'name': 'No artifacts', 'channels': [9]},
    ]
    session = {'root': tmp_path, 'state': {}}
    namespace = {
        'st': fake_st, 'full_session': session, 'session': session, 'groups': groups,
        'selected_observation_group_ids': {1, 2},
        'observation_group_scope_label': lambda group_id: f'Run {group_id}',
        '_load_simulated_sweep_section': lambda *args: True,
        '_pdf_surrogate_file_index': lambda *args: ({}, {1: {10: 'a'}, 2: {20: 'b'}}),
        '_session_for_channel_group': lambda session, group_id: {'id': group_id},
        '_surrogate_files': lambda *args: pytest.fail('Out-of-scope artifacts must not load'),
        **{name: getattr(viewer, name) for name in (
            '_initialize_preference', '_prepare_preferred_multiselect_value',
            '_sync_preferred_widget_value',
        )},
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), viewer.__file__, 'exec'), namespace)
    render = namespace['_render_surrogate_tab']
    assert [group['id'] for group, _, _ in render()] == [1, 2]
    state['bo_surrogate_groups'] = [1]
    assert [group['id'] for group, _, _ in render()] == [1]
    namespace['selected_observation_group_ids'] = {2}
    assert [group['id'] for group, _, _ in render()] == [2]
    assert calls[-1] == ([2], ['Run 2'])
    namespace['selected_observation_group_ids'] = {3}
    assert render() == []
    namespace['selected_observation_group_ids'] = {1, 2, 3}
    assert [group['id'] for group, _, _ in render()] == [1, 2]
