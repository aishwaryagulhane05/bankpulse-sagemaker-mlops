"""Data contract: expected columns, types, allowed values, forbidden columns and business rules.

    contract = load_contract('config/contract.yaml')
    check_schema(df, contract)        columns, types, allowed values, ranges
    check_rules(df, contract)         business rules between columns
    df = drop_forbidden(df, contract) remove leakage columns such as duration
    quality_report(df, contract)      row count, positive rate, missing and "unknown" shares

The checks collect every problem before raising, so one run shows everything that is
wrong with a file instead of one problem at a time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Union

import pandas as pd
import yaml

TYPES = ('int', 'float', 'category')
OPERATORS = ('equals', 'not_equals', 'min')

# How many example bad values an error message shows.
_EXAMPLES = 5


class ContractError(ValueError):
    """The contract file is invalid, or data does not match the contract."""


@dataclass(frozen=True)
class Contract:
    columns: Dict[str, Dict]
    target: str
    positive_label: str
    forbidden_columns: List[str]
    sentinels: Dict[str, object]
    rules: List[Dict]

    @property
    def allowed_columns(self) -> List[str]:
        """Columns features may read: every column except the target and forbidden ones."""
        return [c for c in self.columns if c != self.target and c not in self.forbidden_columns]


# ---- loading -----------------------------------------------------------------------

def _check_condition(rule: str, part: str, condition: Mapping, columns: Mapping) -> None:
    operators = [key for key in condition if key != 'column']
    if condition.get('column') not in columns:
        raise ContractError(f'Rule {rule}: {part} uses unknown column {condition.get("column")!r}')
    if len(operators) != 1 or operators[0] not in OPERATORS:
        raise ContractError(f'Rule {rule}: {part} needs exactly one of {OPERATORS}, got {operators}')


def load_contract(path: Union[str, Path]) -> Contract:
    """Read the contract YAML and check that it is consistent."""
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle) or {}

    # Step 1: required sections.
    for key in ('columns', 'target', 'positive_label'):
        if key not in raw:
            raise ContractError(f'Contract is missing {key}')
    columns = raw['columns']

    # Step 2: every column has a known type; categories list their values. Values are
    # stored as strings, so an unquoted YAML yes/no (read as True/False) is caught here.
    for name, spec in columns.items():
        if spec.get('type') not in TYPES:
            raise ContractError(f'Column {name}: type must be one of {TYPES}, got {spec.get("type")!r}')
        if spec['type'] == 'category':
            values = spec.get('values') or []
            if not values or not all(isinstance(value, str) for value in values):
                raise ContractError(f'Column {name}: category values must be a non-empty list of strings '
                                    f'(quote "yes" and "no"), got {values}')

    # Step 3: target, forbidden columns, sentinels and rules refer to listed columns.
    forbidden = list(raw.get('forbidden_columns') or [])
    sentinels = dict(raw.get('sentinels') or {})
    unknown = [c for c in [raw['target'], *forbidden, *sentinels] if c not in columns]
    if unknown:
        raise ContractError(f'Contract refers to columns not listed under columns: {unknown}')
    rules = list(raw.get('rules') or [])
    for rule in rules:
        _check_condition(rule.get('name'), 'when', rule.get('when') or {}, columns)
        _check_condition(rule.get('name'), 'then', rule.get('then') or {}, columns)

    return Contract(columns=columns, target=raw['target'], positive_label=str(raw['positive_label']),
                    forbidden_columns=forbidden, sentinels=sentinels, rules=rules)


# ---- checks --------------------------------------------------------------------------

def _column_problems(name: str, series: pd.Series, spec: Mapping) -> List[str]:
    """Every way one column breaks its spec, as readable messages."""
    problems = []
    missing = int(series.isna().sum())
    if missing:
        problems.append(f'{name}: {missing} missing values')
    values = series.dropna()

    if spec['type'] == 'category':
        bad = sorted(set(values.astype(str)) - set(spec['values']))
        if bad:
            problems.append(f'{name}: unexpected values {bad[:_EXAMPLES]} (allowed: {spec["values"]})')
        return problems

    # int and float: the dtype must be numeric (bool is not a number here).
    is_int = pd.api.types.is_integer_dtype(series)
    is_number = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if (spec['type'] == 'int' and not is_int) or (spec['type'] == 'float' and not is_number):
        return problems + [f'{name}: expected {spec["type"]}, got {series.dtype}']
    if 'min' in spec and (values < spec['min']).any():
        problems.append(f'{name}: {int((values < spec["min"]).sum())} values below the minimum {spec["min"]}')
    if 'max' in spec and (values > spec['max']).any():
        problems.append(f'{name}: {int((values > spec["max"]).sum())} values above the maximum {spec["max"]}')
    return problems


def _raise_if(problems: List[str], what: str) -> None:
    if problems:
        raise ContractError(f'{what}:\n- ' + '\n- '.join(problems))


def check_schema(df: pd.DataFrame, contract: Contract) -> None:
    """Raise ContractError listing every column, type, value or range problem."""
    # Step 1: the file has exactly the contract's columns.
    problems = []
    missing = [c for c in contract.columns if c not in df.columns]
    extra = [c for c in df.columns if c not in contract.columns]
    if missing:
        problems.append(f'missing columns {missing}')
    if extra:
        problems.append(f'unexpected columns {extra}')

    # Step 2: each present column matches its type, allowed values and range.
    for name, spec in contract.columns.items():
        if name in df.columns:
            problems += _column_problems(name, df[name], spec)
    _raise_if(problems, 'Data does not match the contract')


def _condition(df: pd.DataFrame, condition: Mapping) -> pd.Series:
    """Boolean mask of rows where the condition holds."""
    column = df[condition['column']]
    if 'equals' in condition:
        return column == condition['equals']
    if 'not_equals' in condition:
        return column != condition['not_equals']
    return column >= condition['min']


def check_rules(df: pd.DataFrame, contract: Contract) -> None:
    """Raise ContractError listing every business rule that some rows break."""
    problems = []
    for rule in contract.rules:
        # A rule is broken by rows that match "when" but not "then".
        broken = _condition(df, rule['when']) & ~_condition(df, rule['then'])
        if broken.any():
            problems.append(f'{rule["name"]}: {int(broken.sum())} rows, first at index {broken.idxmax()} '
                            f'({rule.get("description", "")})')
    _raise_if(problems, 'Data breaks business rules')


def drop_forbidden(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """A copy of df without the forbidden (leakage) columns."""
    return df.drop(columns=[c for c in contract.forbidden_columns if c in df.columns])


# ---- reporting -----------------------------------------------------------------------

def quality_report(df: pd.DataFrame, contract: Contract) -> Dict:
    """Summary of the data's size and quality, as JSON-safe values."""
    target = df[contract.target].astype(str)
    categories = [name for name, spec in contract.columns.items()
                  if spec['type'] == 'category' and 'unknown' in spec['values'] and name in df.columns]
    return {
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'duplicate_rows': int(df.duplicated().sum()),
        'positive_rate': round(float((target == contract.positive_label).mean()), 4),
        # Only columns that actually have missing values, to keep the report short.
        'missing_values': {c: int(n) for c, n in df.isna().sum().items() if n},
        'unknown_share': {c: round(float((df[c].astype(str) == 'unknown').mean()), 4) for c in categories},
        'sentinel_share': {c: round(float((df[c] == value).mean()), 4)
                           for c, value in contract.sentinels.items() if c in df.columns},
    }
