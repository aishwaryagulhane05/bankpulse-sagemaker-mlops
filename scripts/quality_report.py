"""Check the raw dataset against the contract and save a quality report.

Run from anywhere:
    python scripts/quality_report.py

Writes <output.reports_dir>/quality_report.json (reports/ by default).
"""

import logging
from pathlib import Path

from bankpulse_core.config import load_config
from bankpulse_core.contract import check_rules, check_schema, load_contract, quality_report
from bankpulse_core.io import file_hash, read_csv, write_json
from bankpulse_core.log_utils import log_stage, setup_logging

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'data_prep.yaml'

logger = logging.getLogger('bankpulse_core.scripts.quality_report')


def main() -> None:
    # Step 1: settings, logging and the contract.
    config = load_config(CONFIG_PATH)
    setup_logging(config['log_level'])
    contract = load_contract(config['contract'])

    # Step 2: read the raw file and check it; any problem stops the script here.
    with log_stage('read raw data'):
        data = read_csv(config['data']['path'], sep=config['data']['separator'])
    with log_stage('check contract'):
        check_schema(data, contract)
        check_rules(data, contract)

    # Step 3: build the report, record which file it describes, and save it.
    with log_stage('quality report'):
        report = {'source': Path(config['data']['path']).name,
                  'source_sha256': file_hash(config['data']['path']),
                  **quality_report(data, contract)}
        path = write_json(report, Path(config['output']['reports_dir']) / 'quality_report.json')
    logger.info('%s rows, %.2f%% positive; saved %s', report['rows'], 100 * report['positive_rate'], path)


if __name__ == '__main__':
    main()
