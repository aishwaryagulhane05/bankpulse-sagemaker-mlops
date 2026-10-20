"""Replay the history as monthly data arrivals, for the continuous-training pipeline (BP-1107).

The dataset is in date order; each change of `month` starts a new calendar month (26 months,
May 2008 - Nov 2010). Run from the repository folder with lab credentials:

    python scripts/simulate_arrivals.py --history 23     # months 1-23 become accepted batches
    python scripts/simulate_arrivals.py --arrive 24      # month 24 lands in raw/incoming/

--arrive uploads one file to raw/incoming/; with the S3 trigger (BP-1105) that upload starts the
retraining pipeline by itself. Only new objects are written; nothing is deleted or overwritten.
"""

import argparse
import tempfile
from pathlib import Path

from bankpulse_core.aws import lab_boto_session, read_state, s3_uri
from bankpulse_core.config import load_config
from bankpulse_core.contract import load_contract
from bankpulse_core.ingest import ingest_file, month_batches, read_manifest
from bankpulse_core.io import read_csv, upload_file

ROOT = Path(__file__).resolve().parents[1]


def month_file(months, number: int, folder: Path) -> Path:
    """Month `number` (1-based) written exactly like a raw file: ';'-separated, with a header."""
    path = folder / f'month-{number:02d}.csv'
    months[number - 1].to_csv(path, sep=';', index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--history', type=int, help='accept months 1..N directly as batches')
    group.add_argument('--arrive', type=int, help='upload month N to raw/incoming/')
    args = parser.parse_args()

    config = load_config(ROOT / 'config' / 'data_prep.yaml')
    state = read_state(Path(config['output']['processed_dir']) / 'sagemaker_state.json')
    lab_boto_session()                                      # bankpulse-lab profile for every S3 call
    batches = s3_uri(state['bucket'], state['prefix'], 'raw', 'batches')
    months = month_batches(read_csv(config['data']['path'], sep=config['data']['separator']))
    folder = Path(tempfile.mkdtemp(prefix='bankpulse-months-'))

    if args.history:
        contract = load_contract(config['contract'])
        for number in range(1, args.history + 1):
            result = ingest_file(month_file(months, number, folder), contract, batches,
                                 separator=config['data']['separator'])
            print(f'month {number:2d}: {result["status"]}')
        print(len(read_manifest(batches)['batches']), 'batches in', batches)
    else:
        uri = upload_file(month_file(months, args.arrive, folder),
                          s3_uri(state['bucket'], state['prefix'], 'raw', 'incoming', f'month-{args.arrive:02d}.csv'))
        print('arrived:', uri)


if __name__ == '__main__':
    main()
