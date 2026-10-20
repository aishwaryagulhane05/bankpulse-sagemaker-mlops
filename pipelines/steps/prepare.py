"""Pipeline step 1 (Processing): ingest new files, take the rolling window, build the training files.

Runs inside a SageMaker Processing job, with bankpulse_core, features and config/ next to it:
    python prepare.py --batches s3://.../raw/batches --incoming s3://.../raw/incoming/ \
                      --window 12,2,2 --capped-campaign-version v1

Writes to /opt/ml/processing/output (uploaded to S3 by SageMaker when the job ends):
    train/ validation/ batch-input/ labels/ feature-schema/ run_metadata.json ingest_report.json
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The bundle puts bankpulse_core/ and features/ next to this script; make them importable.
sys.path.insert(0, str(HERE))

from bankpulse_core.config import load_config  # noqa: E402
from bankpulse_core.contract import load_contract  # noqa: E402
from bankpulse_core.ingest import ingest_file  # noqa: E402
from bankpulse_core.io import is_s3, list_objects, write_json  # noqa: E402
from bankpulse_core.pipeline import run  # noqa: E402

OUTPUT = Path('/opt/ml/processing/output')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', required=True, help='S3 prefix with manifest.json and the accepted batches')
    parser.add_argument('--incoming', required=True, help='S3 prefix where new monthly files land')
    parser.add_argument('--window', default='12,2,2', help='train,validation,test batch counts')
    parser.add_argument('--capped-campaign-version', default='v1')
    parser.add_argument('--output', default=str(OUTPUT))
    return parser.parse_args(argv)


def waiting_files(incoming: str) -> list:
    """CSV files in the incoming prefix (S3) or folder (local runs and tests), oldest name first."""
    if is_s3(incoming):
        return [uri for uri in list_objects(incoming) if uri.endswith('.csv')]
    return sorted(str(path) for path in Path(incoming).glob('*.csv'))


def main(argv=None) -> dict:
    args = parse_args(argv)
    config = load_config(HERE / 'config' / 'data_prep.yaml')
    contract = load_contract(config['contract'])

    # Step 1: ingest every file waiting in incoming/. Files already in the manifest are skipped
    # (duplicates), invalid ones are recorded under rejected/; nothing is deleted.
    incoming = waiting_files(args.incoming)
    report = [ingest_file(uri, contract, args.batches, separator=config['data']['separator']) for uri in incoming]
    for item in report:
        print(f"ingest {item['status']:9s} {item['source']}")

    # Step 2: the rolling window over all accepted batches, and the feature versions for this run.
    train, validation, test = (int(n) for n in args.window.split(','))
    config['data']['batches'] = args.batches
    config['window'] = {'train_batches': train, 'validation_batches': validation, 'test_batches': test}
    config['features']['capped_campaign'] = args.capped_campaign_version
    config['output']['processed_dir'] = args.output
    config['output']['reports_dir'] = str(Path(args.output) / 'reports')

    # Step 3: the same pipeline as notebook 01, so the files are in the same XGBoost format.
    result = run(config)
    write_json({'ingested': report, 'split_summary': result['split_summary'], 'n_columns': result['n_columns']},
               Path(args.output) / 'ingest_report.json')
    print('split summary:', result['split_summary'])
    return result


if __name__ == '__main__':
    main()
