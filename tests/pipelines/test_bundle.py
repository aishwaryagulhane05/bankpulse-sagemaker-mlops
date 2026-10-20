"""The Processing-job bundle contains everything the step scripts import."""

from pipelines.retrain_pipeline import build_bundle


def test_bundle_has_framework_features_config_and_steps(tmp_path):
    bundle = build_bundle(tmp_path / 'bundle')
    for path in ('bankpulse_core/pipeline.py', 'bankpulse_core/ingest.py', 'features/campaign.py',
                 'config/data_prep.yaml', 'config/contract.yaml', 'prepare.py', 'evaluate.py'):
        assert (bundle / path).is_file(), path
    # pandas 2.0.3: the newest release built for the container's NumPy 1.24.
    assert 'pandas==2.0.3' in (bundle / 'requirements.txt').read_text()
    assert not list(bundle.rglob('__pycache__'))
