from types import SimpleNamespace

import pandas as pd
import pytest

import bo_session_viewer as viewer


@pytest.mark.parametrize('interval, expected', [(None, [5]), (2, [1, 3, 5]), (3, [1, 4, 5]), (1, [1, 2, 3, 4, 5]), (20, [1, 5])])
def test_snapshot_export_keeps_full_history_and_both_surfaces(tmp_path, monkeypatch, interval, expected):
    history = pd.DataFrame({
        'iteration': [1, 2, 3, 4, 5],
        'frequency': [100.0] * 5,
        'amplitude': [0.03] * 5,
        'step_potential': [0.002] * 5,
        'observed_value': [1., 2., 3., 4., 5.],
    })
    original = history.copy(deep=True)
    ground_truth = pd.DataFrame({'fitness': [1.]})
    candidate_pool = pd.DataFrame({'fitness': [2.]})
    histories = []

    def artifact(source, partial_history, axes, *, config, prepared_candidates, model_out):
        assert source is candidate_pool
        histories.append(partial_history['iteration'].tolist())
        model_out["gp"] = SimpleNamespace(kernel_="test", log_marginal_likelihood_value_=0.)
        return pd.DataFrame({'predicted_mean_Q': [partial_history['observed_value'].mean()], 'acquisition_value': [0.5]})

    monkeypatch.setattr(viewer, '_simulation_surrogate_artifact_frame', artifact)
    monkeypatch.setattr(viewer, '_simulation_fit_exportable_gp', lambda *args, **kwargs: SimpleNamespace(kernel_='test', log_marginal_likelihood_value_=0.))
    progress = []
    viewer._write_simulated_surrogate_artifacts(
        tmp_path, ground_truth, history,
        axes=('frequency', 'amplitude', 'step_potential'), group_id=1,
        config={}, candidate_pool=candidate_pool,
        surrogate_snapshot_interval=interval,
        progress_callback=lambda fraction, message: progress.append(fraction),
    )
    assert histories == [list(range(1, iteration + 1)) for iteration in expected]
    assert len(list((tmp_path / 'surrogate').glob('*_candidate_predictions.csv'))) == len(expected)
    for iteration in expected:
        stem = f'group_01_iter_{iteration:03d}'
        prediction = tmp_path / 'surrogate' / f'{stem}_candidate_predictions.csv'
        acquisition = tmp_path / 'acquisition' / f'{stem}_acquisition_values.csv'
        assert prediction.read_bytes() == acquisition.read_bytes()
        assert set(pd.read_csv(prediction)) == {'predicted_mean_Q', 'acquisition_value'}
    pd.testing.assert_frame_equal(history, original)
    assert progress[-1] == 1.0


def test_export_prepares_grid_once_and_fits_once_per_snapshot(tmp_path, monkeypatch):
    import pickle
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor

    axes = ('frequency', 'amplitude', 'step_potential')
    candidates = pd.DataFrame({
        'frequency': [100., 150., 200.], 'amplitude': [.03] * 3,
        'step_potential': [.002] * 3, 'ground_truth_value': [1., 2., 3.],
    })
    history = candidates.rename(columns={'ground_truth_value': 'observed_value'}).assign(iteration=[1, 2, 3])
    config = {'parameters': {'frequency': {'mode': 'variable', 'values': [100., 150., 200.]}}}
    prepared_calls = []
    fit_calls = []
    prepare = viewer._prepare_simulation_surrogate_candidates
    fit = GaussianProcessRegressor.fit

    def counted_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        prepared_calls.append(prepared)
        return prepared

    def counted_fit(self, x, y):
        fit_calls.append(len(y))
        return fit(self, x, y)

    monkeypatch.setattr(viewer, '_prepare_simulation_surrogate_candidates', counted_prepare)
    monkeypatch.setattr(GaussianProcessRegressor, 'fit', counted_fit)
    viewer._write_simulated_surrogate_artifacts(
        tmp_path, candidates, history, axes=axes, group_id=1, config=config,
    )
    assert len(prepared_calls) == 1
    assert fit_calls == [2, 3]
    with (tmp_path / 'surrogate/group_01_iter_003_gp_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    means, stds = model.predict(prepared_calls[0]['encoded'], return_std=True)
    saved = pd.read_csv(tmp_path / 'surrogate/group_01_iter_003_candidate_predictions.csv')
    np.testing.assert_allclose(saved['predicted_mean_Q'], means)
    np.testing.assert_allclose(saved['predicted_std_Q'], stds)


def test_numpy_fallback_still_writes_prediction_artifacts(tmp_path, monkeypatch):
    import json
    from sklearn.gaussian_process import GaussianProcessRegressor

    def failed_fit(*args, **kwargs):
        raise RuntimeError('test sklearn failure')

    monkeypatch.setattr(GaussianProcessRegressor, 'fit', failed_fit)
    candidates = pd.DataFrame({
        'frequency': [100., 200.], 'amplitude': [.03, .03],
        'step_potential': [.002, .002], 'ground_truth_value': [1., 2.],
    })
    history = candidates.rename(columns={'ground_truth_value': 'observed_value'}).assign(iteration=[1, 2])
    viewer._write_simulated_surrogate_artifacts(
        tmp_path, candidates, history, axes=('frequency', 'amplitude', 'step_potential'),
        group_id=1, config={}, surrogate_snapshot_interval=None,
    )
    metadata = json.loads((tmp_path / 'surrogate/group_01_iter_002_surrogate_metadata.json').read_text())
    assert metadata['backend'] == 'numpy_gaussian_process'
    assert metadata['gp_pickle_error'] == 'test sklearn failure'
    assert (tmp_path / 'surrogate/group_01_iter_002_candidate_predictions.csv').is_file()
    assert not (tmp_path / 'surrogate/group_01_iter_002_gp_model.pkl').exists()
