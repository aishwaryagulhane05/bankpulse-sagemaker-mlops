"""SageMaker Pipeline "bankpulse-retrain": continuous training while the endpoint keeps serving.

    python -m pipelines.retrain_pipeline            # create or update the pipeline definition
    python -m pipelines.retrain_pipeline --start    # ... and start one execution

Steps (docs/retraining.md):
    Prepare   Processing: ingest raw/incoming/, rolling window, pipeline.run -> XGBoost files
    Train     Training job: built-in XGBoost 1.7-1, tuned settings from reports/xgboost_tuning.json
    Evaluate  Processing (XGBoost container): challenger vs champion on the newest holdout batches
    Check     Condition: challenger_ok == 1
    Register  Model Registry, PendingManualApproval (a person approves; nothing is deployed here)

The pipeline only adds things: new batches, new jobs, new model versions. It never changes or
deletes the deployed endpoint, the registered versions or the existing data lake files.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost import XGBoost

from bankpulse_core.aws import lab_boto_session, read_state, s3_uri
from bankpulse_core.config import load_config
from bankpulse_core.io import read_json

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_NAME = 'bankpulse-retrain'
XGBOOST_VERSION = '1.7-1'
INSTANCE = 'ml.m5.large'
PROC_BASE = '/opt/ml/processing'


def build_bundle(target: Path) -> Path:
    """The code the Processing jobs run: framework, features, config and the step scripts, flat."""
    target.mkdir(parents=True, exist_ok=True)
    for package in ('bankpulse_core', 'features', 'config'):
        shutil.copytree(ROOT / package, target / package, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for script in (ROOT / 'pipelines' / 'steps').glob('*.py'):
        if script.name != '__init__.py':
            shutil.copy(script, target / script.name)
    # The XGBoost 1.7-1 container (Python 3.9) ships pandas 1.2.4, too old for this code, and
    # NumPy 1.24. pandas 2.0.3 is the newest release built for NumPy 1.x; newer wheels need NumPy 2
    # and fail to import ("numpy.core.multiarray failed to import").
    (target / 'requirements.txt').write_text('pyyaml>=5.4\npandas==2.0.3\n', encoding='utf-8')
    return target


def build_pipeline(session: PipelineSession, role: str, bucket: str, prefix: str, champion: dict) -> Pipeline:
    lake = s3_uri(bucket, prefix)
    tuning = read_json(ROOT / 'reports' / 'xgboost_tuning.json')
    best = tuning['best']

    # ---- parameters: what can change per execution -----------------------------------------------
    window = ParameterString('Window', default_value='12,2,2')
    capped_version = ParameterString('CappedCampaignVersion', default_value='v1')
    margin = ParameterFloat('ApprovalMargin', default_value=0.01)
    champion_model = ParameterString('ChampionModelUri', default_value=champion['model_data'])
    champion_schema = ParameterString('ChampionSchemaUri', default_value=champion['schema_dir'])
    batches = s3_uri(bucket, prefix, 'raw', 'batches')
    incoming = s3_uri(bucket, prefix, 'raw', 'incoming') + '/'

    bundle = build_bundle(Path(tempfile.mkdtemp(prefix='bankpulse-bundle-')))
    processor = FrameworkProcessor(estimator_cls=XGBoost, framework_version=XGBOOST_VERSION, py_version='py3',
                                   role=role, instance_type=INSTANCE, instance_count=1,
                                   base_job_name='bankpulse-retrain', sagemaker_session=session)

    # ---- 1. Prepare -----------------------------------------------------------------------------
    prepare = ProcessingStep('Prepare', step_args=processor.run(
        code='prepare.py', source_dir=str(bundle),
        outputs=[ProcessingOutput(output_name='processed', source=f'{PROC_BASE}/output',
                                  destination=Join('/', [lake, 'retraining', 'executions',
                                                         ExecutionVariables.PIPELINE_EXECUTION_ID, 'processed']))],
        arguments=['--batches', batches, '--incoming', incoming, '--window', window,
                   '--capped-campaign-version', capped_version]))
    processed = prepare.properties.ProcessingOutputConfig.Outputs['processed'].S3Output.S3Uri

    # ---- 2. Train the challenger ----------------------------------------------------------------
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve('xgboost', region=session.boto_region_name, version=XGBOOST_VERSION),
        role=role, instance_count=1, instance_type=INSTANCE, max_run=3600, disable_profiler=True,
        debugger_hook_config=False, base_job_name='bankpulse-retrain-xgboost',
        output_path=s3_uri(bucket, prefix, 'retraining', 'model-artifacts'), sagemaker_session=session,
        hyperparameters={'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'num_round': tuning['num_round'],
                         'early_stopping_rounds': tuning['early_stopping_rounds'], 'max_depth': best['max_depth'],
                         'eta': best['eta'], 'min_child_weight': best['min_child_weight'],
                         'subsample': tuning['fixed']['subsample'], 'colsample_bytree': best['colsample_bytree']})
    train = TrainingStep('Train', step_args=estimator.fit({
        'train': TrainingInput(Join('/', [processed, 'train/']), content_type='text/csv'),
        'validation': TrainingInput(Join('/', [processed, 'validation/']), content_type='text/csv'),
    }))

    # ---- 3. Evaluate challenger vs champion ---------------------------------------------------------
    evaluation_file = PropertyFile(name='Evaluation', output_name='evaluation', path='evaluation.json')
    evaluate = ProcessingStep('Evaluate', property_files=[evaluation_file], step_args=processor.run(
        code='evaluate.py', source_dir=str(bundle),
        inputs=[
            ProcessingInput(source=train.properties.ModelArtifacts.S3ModelArtifacts,
                            destination=f'{PROC_BASE}/challenger/model'),
            ProcessingInput(source=Join('/', [processed, 'feature-schema']), destination=f'{PROC_BASE}/challenger/schema'),
            ProcessingInput(source=champion_model, destination=f'{PROC_BASE}/champion/model'),
            ProcessingInput(source=champion_schema, destination=f'{PROC_BASE}/champion/schema'),
        ],
        outputs=[ProcessingOutput(output_name='evaluation', source=f'{PROC_BASE}/evaluation')],
        arguments=['--batches', batches, '--window', window, '--margin', margin.to_string()]))
    evaluation_uri = evaluate.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri

    # ---- 4. Register only a challenger that passes ------------------------------------------------
    model = Model(image_uri=estimator.image_uri, model_data=train.properties.ModelArtifacts.S3ModelArtifacts,
                  role=role, sagemaker_session=session)
    register = ModelStep('Register', step_args=model.register(
        model_package_group_name=champion['group'], approval_status='PendingManualApproval',
        content_types=['text/csv'], response_types=['text/csv'],
        inference_instances=[INSTANCE], transform_instances=[INSTANCE],
        model_metrics=ModelMetrics(model_statistics=MetricsSource(
            s3_uri=Join('/', [evaluation_uri, 'model_quality.json']), content_type='application/json')),
        description='Retrained by bankpulse-retrain on the rolling window; see evaluation.json',
        customer_metadata_properties={'feature_schema_uri': Join('/', [processed, 'feature-schema', 'feature_schema.json']),
                                      'evaluation_uri': Join('/', [evaluation_uri, 'evaluation.json'])},
    ))
    check = ConditionStep('CheckChallenger', conditions=[ConditionEquals(
        left=JsonGet(step_name=evaluate.name, property_file=evaluation_file, json_path='challenger_ok'), right=1)],
        if_steps=[register], else_steps=[])

    return Pipeline(name=PIPELINE_NAME, parameters=[window, capped_version, margin, champion_model, champion_schema],
                    steps=[prepare, train, evaluate, check], sagemaker_session=session)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true', help='start an execution after updating the definition')
    parser.add_argument('--capped-campaign-version', default=None, help='override the parameter for this execution')
    args = parser.parse_args()

    config = load_config(ROOT / 'config' / 'data_prep.yaml')
    state = read_state(Path(config['output']['processed_dir']) / 'sagemaker_state.json')
    boto_session = lab_boto_session()
    session = PipelineSession(boto_session=boto_session)
    champion = {'model_data': state['model_data'], 'group': state['model_package_group'],
                'schema_dir': state['s3']['feature_schema'].rsplit('/', 1)[0]}

    pipeline = build_pipeline(session, state['role'], state['bucket'], state['prefix'], champion)
    pipeline.upsert(role_arn=state['role'], description='BankPulse continuous training (docs/retraining.md)')
    print('pipeline', PIPELINE_NAME, 'created/updated')
    if args.start:
        parameters = {'CappedCampaignVersion': args.capped_campaign_version} if args.capped_campaign_version else {}
        execution = pipeline.start(parameters=parameters)
        print('started', execution.arn)


if __name__ == '__main__':
    main()
