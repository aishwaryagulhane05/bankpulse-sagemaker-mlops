"""Score customers with the live endpoint, encoding them exactly like the training data."""

import re
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from bankpulse_core.pipeline import encode_records, to_csv_payload

# Fields an agent knows before calling, in form order (no duration, no outcome).
FORM_FIELDS = ('age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
               'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome')

# Requests per InvokeEndpoint call: small payloads, and Data Capture logs each call.
BATCH = 50


def form_options(contract) -> Dict[str, List[str]]:
    """Allowed values for every category field, from the data contract."""
    return {name: list(spec['values']) for name, spec in contract.columns.items()
            if spec['type'] == 'category' and name in FORM_FIELDS}


def required_inputs(schema: Mapping) -> List[str]:
    """Raw columns the deployed model's features read, from feature_schema.json."""
    return sorted({column for feature in schema['features'] for column in feature['inputs']})


def parse_scores(body: str) -> List[float]:
    """Probabilities from an XGBoost text/csv response (newline- or comma-separated)."""
    return [float(value) for value in re.split(r'[,\n]', body) if value.strip()]


def score(runtime, endpoint_name: str, records: Sequence[Mapping], schema: Mapping) -> List[float]:
    """One probability per record, from the endpoint."""
    rows = encode_records(list(records), schema)
    scores: List[float] = []
    for start in range(0, len(rows), BATCH):
        response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Accept='text/csv',
                                           Body=to_csv_payload(rows.iloc[start:start + BATCH]))
        scores += parse_scores(response['Body'].read().decode('utf-8'))
    if len(scores) != len(rows):
        raise ValueError(f'endpoint returned {len(scores)} scores for {len(rows)} customers')
    return scores


def rank(customers: pd.DataFrame, scores: Sequence[float], threshold: float) -> pd.DataFrame:
    """Customers with score, call-first flag and rank (1 = call first)."""
    ranked = customers.assign(score=[round(s, 4) for s in scores])
    ranked['call_first'] = ranked['score'] >= threshold
    ranked['rank'] = ranked['score'].rank(ascending=False, method='first').astype(int)
    return ranked.sort_values('rank')
