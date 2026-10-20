"""BankPulse results UI.

Run from the repository folder:
    streamlit run ui/app.py

Sections: project overview, live SageMaker tracking, model results, monitoring, and scoring customers
with the real-time endpoint. Works offline from the saved reports; with lab credentials in
~/.aws (profile bankpulse-lab) it also reads live status from SageMaker, S3 and CloudWatch.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from bankpulse_core.aws import lab_boto_session, read_state
from bankpulse_core.config import load_config
from bankpulse_core.contract import load_contract
from bankpulse_core.io import read_csv, read_json
from ui import content, scoring, tracking

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
METRICS = ['positive_rate', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'lift_top_decile']
SECTIONS = ['Overview', 'SageMaker tracking', 'Model results', 'Monitoring', 'Score customers']
COLOURS = {'alert': '#C0392B', 'warn': '#E07B00', 'stable': '#2E8B57'}

st.set_page_config(page_title='BankPulse | SageMaker results', page_icon='📞', layout='wide')


# ---- data loading (cached) ----------------------------------------------------------------------

@st.cache_data
def load_local():
    """Config, contract, saved reports and the state written by the notebooks."""
    config = load_config(ROOT / 'config' / 'data_prep.yaml')
    processed = Path(config['output']['processed_dir'])
    reports = {path.stem: read_json(path) for path in REPORTS.glob('*.json')}
    schema_path = processed / 'feature-schema' / 'feature_schema.json'
    return {'config': config, 'reports': reports, 'state': read_state(processed / 'sagemaker_state.json'),
            'schema': read_json(schema_path) if schema_path.exists() else None}


@st.cache_resource
def aws():
    """boto3 session for the lab, or None when no valid credentials are available."""
    session = lab_boto_session()
    identity = tracking.safe(lambda: session.client('sts').get_caller_identity())
    if 'error' in identity:
        return None, identity['error']
    return session, f'{identity["Arn"].split("/")[-2]} · {session.region_name}'


@st.cache_data(ttl=60, show_spinner='Reading SageMaker status...')
def live_status(_session, _state, state_key):
    """All live statuses at once; cached for a minute. state_key (JSON of the state) is the cache key."""
    state = _state
    sm, s3, cw = _session.client('sagemaker'), _session.client('s3'), _session.client('cloudwatch')
    now = datetime.now(timezone.utc)
    return {
        'data_lake': tracking.safe(lambda: tracking.data_lake(s3, state['bucket'], state['prefix'])),
        'training': tracking.safe(lambda: tracking.training_job(sm, state['training_job'])),
        'transforms': tracking.safe(lambda: tracking.transform_jobs(sm, state['transform_jobs'])),
        'registry': tracking.safe(lambda: tracking.registry(sm, state['model_package_group'])),
        'model_card': tracking.safe(lambda: tracking.model_card(sm, state['model_card'])),
        'endpoint': tracking.safe(lambda: tracking.endpoint(sm, state['endpoint'])),
        'alarm': tracking.safe(lambda: tracking.alarm(cw, state['cloudwatch']['alarm'])),
        'metrics': tracking.safe(lambda: tracking.latest_metrics(
            cw, state['cloudwatch']['namespace'], state['endpoint'],
            ['max_feature_psi', 'prediction_psi', 'positive_rate', 'subgroup_min_max_ratio'],
            now - timedelta(days=14), now)),
    }


@st.cache_data(show_spinner='Loading recent customers...')
def recent_customers(path, separator, n):
    raw = read_csv(path, sep=separator)
    return raw.iloc[-n:].reset_index(drop=True)


def console(region, fragment):
    return f'https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/{fragment}'


def show_error(result, what):
    if isinstance(result, dict) and 'error' in result:
        st.warning(f'{what}: unavailable ({result["error"][:160]})')
        return True
    return False


# ---- page --------------------------------------------------------------------------------------

local = load_local()
reports, state, config = local['reports'], local['state'], local['config']
session, who = aws()

with st.sidebar:
    st.header('📞 BankPulse')
    st.caption('Term deposit propensity model on AWS SageMaker')
    # Sections live in the sidebar (kept across reruns, unlike tabs, so a form result stays on screen).
    page = st.radio('Section', SECTIONS, key='section')
    st.divider()
    if session:
        st.success(f'Connected to the AWS lab\n\n{who}')
    else:
        st.info('Offline: showing saved reports.\n\nTo see live SageMaker status, start the lab, refresh '
                '`~/.aws/credentials` ([bankpulse-lab]) and restart the app.')
    if st.button('Refresh live status'):
        live_status.clear()
    st.divider()
    st.caption('Code: github.com/aishwaryagulhane05/bankpulse-sagemaker-mlops')

LIVE_READY = bool(session and state.get('training_job'))


def get_live():
    return live_status(session, state, json.dumps(state, sort_keys=True))



# ---- Overview ------------------------------------------------------------------------------------
if page == 'Overview':
    st.title(content.TITLE)
    st.markdown(content.TAGLINE)
    xgb, base = reports['xgboost_metrics']['test'], reports['baseline_metrics']['test']
    extra = xgb['confusion_matrix']['tp'] / base['confusion_matrix']['tp'] - 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Calls in the dataset', f'{reports["quality_report"]["rows"]:,}')
    c2.metric('Subscribe without targeting', f'{reports["quality_report"]["positive_rate"]:.1%}')
    c3.metric('Extra subscribers found vs baseline', f'+{extra:.0%}', help='True positives on the test period, '
              'at the threshold chosen on validation')
    c4.metric('Top 10% lift (test)', f'{xgb["lift_top_decile"]:.2f}x')

    left, right = st.columns(2)
    with left:
        st.subheader('The problem')
        st.markdown(content.PROBLEM)
        st.subheader('How it helps the end user')
        st.markdown(content.HOW_IT_HELPS)
    with right:
        st.subheader('Why it matters')
        st.markdown(content.WHY_IT_MATTERS)
        st.warning(content.NOT_FOR, icon='⚠️')

    st.subheader('Who it is for')
    st.dataframe(pd.DataFrame(content.AUDIENCE, columns=['Audience', 'Role', 'What they get']), hide_index=True)
    st.subheader('How it is built')
    st.dataframe(pd.DataFrame(content.LIFECYCLE, columns=['Stage', 'AWS component', 'What it does']), hide_index=True)
    st.image(str(ROOT / 'architecture' / 'architecture.png'))

# ---- SageMaker tracking -----------------------------------------------------------------------------
if page == 'SageMaker tracking':
    st.header('SageMaker tracking')
    if not state:
        st.info('No lab state yet: run notebooks 01-05 to create the SageMaker resources.')
    elif not LIVE_READY:
        st.info('Offline. These are the resources the notebooks created; connect to the lab for live status.')
        st.json({k: state[k] for k in ('training_job', 'transform_jobs', 'model_package_group', 'model_card',
                                       'endpoint') if k in state})
    else:
        region = session.region_name
        live = get_live()
        st.caption(f'Live, read-only status (refreshed every minute) · region {region}')

        st.subheader('1. Data lake (S3)')
        if not show_error(live['data_lake'], 'S3'):
            lake = pd.DataFrame(live['data_lake']).T.rename_axis('folder').reset_index()
            lake['last_modified'] = lake['last_modified'].astype(str).str.slice(0, 19)
            st.dataframe(lake, hide_index=True)
            st.link_button('Open in S3', f'https://s3.console.aws.amazon.com/s3/buckets/{state["bucket"]}'
                                         f'?region={region}&prefix={state["prefix"]}/')

        st.subheader('2. Training job')
        job = live['training']
        if not show_error(job, 'Training job'):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Status', job['status'])
            c2.metric('Instance', job['instance'])
            c3.metric('Billable seconds', job['billable_seconds'])
            c4.metric('Validation PR-AUC', job['metrics'].get('validation:aucpr'))
            with st.expander('Hyperparameters and artifact'):
                st.json(job['hyperparameters'])
                st.code(job['model_artifact'])
            st.link_button('Open training job', console(region, f'jobs/{job["name"]}'))

        st.subheader('3. Batch Transform')
        if not show_error(live['transforms'], 'Batch Transform'):
            st.dataframe(pd.DataFrame(live['transforms']).T.rename_axis('split').reset_index(), hide_index=True)

        st.subheader('4. Model Registry and Model Card')
        if not show_error(live['registry'], 'Model Registry'):
            versions = pd.DataFrame(live['registry'])
            versions['created'] = versions['created'].astype(str).str.slice(0, 19)
            st.dataframe(versions[['version', 'status', 'created', 'description', 'approval_note']], hide_index=True)
        if not show_error(live['model_card'], 'Model Card'):
            card = live['model_card']
            st.write(f'Model Card **{card["name"]}**: {card["status"]} (version {card["version"]})')

        st.subheader('5. Endpoint and Data Capture')
        ep = live['endpoint']
        if not show_error(ep, 'Endpoint'):
            c1, c2, c3 = st.columns(3)
            c1.metric('Endpoint status', ep['status'])
            if ep['status'] != 'Deleted':
                c2.metric('Data Capture', f'{ep["capture_percent"]}%' if ep['data_capture'] else 'off')
                c3.metric('Instances', sum(ep['instance']))
                st.link_button('Open endpoint', console(region, f'endpoints/{ep["name"]}'))
                st.warning('The endpoint bills every hour it runs. Delete it with the clean-up cell in notebook 05 '
                           'when finished.', icon='💰')

# ---- Model results ------------------------------------------------------------------------------
if page == 'Model results':
    st.header('Model results')
    st.caption('Same 68 inputs for both models. Threshold chosen on validation; test evaluated once.')
    baseline, xgboost = reports['baseline_metrics'], reports['xgboost_metrics']
    split = st.radio('Split', ['test', 'validation'], horizontal=True)
    cols = st.columns(4)
    for col, metric in zip(cols, ['pr_auc', 'f1', 'recall', 'lift_top_decile']):
        col.metric(f'XGBoost {metric}', xgboost[split][metric],
                   delta=round(xgboost[split][metric] - baseline[split][metric], 4), help='Delta vs the baseline')

    table = pd.DataFrame({'Logistic baseline': {m: baseline[split][m] for m in METRICS},
                          'SageMaker XGBoost': {m: xgboost[split][m] for m in METRICS}})
    st.dataframe(table)
    long = table.drop(index='positive_rate').reset_index(names='metric').melt('metric', var_name='model')
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X('model:N', title=None, axis=alt.Axis(labels=False)), y='value:Q', color='model:N',
        column=alt.Column('metric:N', title=None)).properties(width=90, height=220))

    st.subheader('Confusion matrices at the threshold')
    left, right = st.columns(2)
    for col, (name, model) in zip((left, right), (('Logistic baseline', baseline), ('SageMaker XGBoost', xgboost))):
        cm = model[split]['confusion_matrix']
        col.write(f'**{name}** (threshold {model[split]["threshold"]})')
        col.dataframe(pd.DataFrame([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]],
                                   index=['actual no', 'actual yes'], columns=['predicted no', 'predicted yes']))

    with st.expander('How the hyperparameters were chosen (validation only)'):
        tuning = reports['xgboost_tuning']
        st.write('Best:', tuning['best'])
        st.dataframe(pd.DataFrame(tuning['top10']), hide_index=True)
    st.info('The test period subscribes far more often (38% vs 6% in training), so compare models on the same split, '
            'not across splits. See the Monitoring tab for the drift.')

# ---- Monitoring -----------------------------------------------------------------------------------
if page == 'Monitoring':
    st.header('Monitoring: live traffic vs training data')
    report = reports['monitoring_report']
    m = report['metrics']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Max feature PSI', m['max_feature_psi'], help='Above 0.2 = significant drift')
    c2.metric('Prediction PSI', m['prediction_psi'])
    c3.metric('Share above threshold', f'{m["positive_rate"]:.0%}')
    c4.metric('Age-band selection ratio', m['subgroup_min_max_ratio'], help='min/max; below 0.8 needs review')
    for alert_text in report['alerts']:
        st.error(alert_text, icon='🚨')
    st.caption(f'Report generated {report["generated_at"][:19]} UTC from {report["records"]} captured requests.')

    left, right = st.columns([3, 2])
    with left:
        psi = pd.Series(report['data_drift']['psi']).head(15).rename_axis('feature').reset_index(name='psi')
        psi['status'] = psi['psi'].map(lambda v: 'alert' if v > 0.2 else 'warn' if v > 0.1 else 'stable')
        st.altair_chart(alt.Chart(psi, title='Feature drift (PSI), top 15').mark_bar().encode(
            x='psi:Q', y=alt.Y('feature:N', sort='-x', title=None),
            color=alt.Color('status:N', scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())))))
    with right:
        groups = pd.DataFrame(report['subgroups']['groups']).T.rename_axis('age band').reset_index()
        st.altair_chart(alt.Chart(groups, title='Share the model would call, by age band').mark_bar().encode(
            x=alt.X('age band:N', sort=['lt30', '30_44', '45_59', '60plus']),
            y=alt.Y('selected_rate:Q', axis=alt.Axis(format='%'), title='selected'), tooltip=['rows']))
        drift = report['prediction_drift']
        st.write(f'Mean score: **{drift["reference_mean"]}** on validation -> **{drift["current_mean"]}** live')

    st.subheader('CloudWatch (live)')
    if LIVE_READY and state.get('cloudwatch'):
        live = get_live()
        if not show_error(live['alarm'], 'Alarm'):
            st.write(f'Alarm **{live["alarm"]["name"]}**: `{live["alarm"]["state"]}`')
            st.caption('Each monitoring run publishes one value. Between runs there is no data, which the alarm '
                       'treats as not breaching, so it shows OK until the next run reports drift above 0.2.')
        if not show_error(live['metrics'], 'Metrics'):
            st.dataframe(pd.Series(live['metrics'], name='latest value').to_frame())
        region = session.region_name
        st.link_button('Open CloudWatch dashboard', f'https://{region}.console.aws.amazon.com/cloudwatch/home'
                                                    f'?region={region}#dashboards/dashboard/{state["cloudwatch"]["dashboard"]}')
    else:
        st.info('Connect to the lab to see the alarm state and the latest metric values.')
    attempts = reports.get('managed_monitoring_attempts', {})
    with st.expander('Managed SageMaker monitoring (Clarify, Model Monitor) in this lab'):
        st.json(attempts)

# ---- Score customers ----------------------------------------------------------------------------------
if page == 'Score customers':
    st.header('Score customers with the live endpoint')
    schema, threshold = local['schema'], float(state.get('threshold', reports['xgboost_metrics']['threshold']))
    endpoint_name = state.get('endpoint')
    status = None
    if session and endpoint_name:
        ep = tracking.safe(lambda: tracking.endpoint(session.client('sagemaker'), endpoint_name))
        status = ep.get('status') if isinstance(ep, dict) else None
    if schema is None:
        st.warning('No feature schema: run `python scripts/baseline.py` or notebook 01 first.')
    elif status != 'InService':
        st.warning(f'Endpoint not available (status: {status or "not connected"}). Start the lab and deploy it '
                   'with notebook 04, then refresh.')
    else:
        runtime = session.client('sagemaker-runtime')
        st.caption(f'Endpoint `{endpoint_name}` · threshold {threshold:.3f} (chosen on validation) · '
                   'only information known before the call is used')

        st.subheader('One customer')
        options = scoring.form_options(load_contract(config['contract']))
        with st.form('customer'):
            c1, c2, c3, c4 = st.columns(4)
            record = {
                'age': c1.number_input('Age', 17, 100, 35),
                'job': c2.selectbox('Job', options['job'], index=options['job'].index('admin.')),
                'marital': c3.selectbox('Marital status', options['marital'], index=1),
                'education': c4.selectbox('Education', options['education'], index=6),
                'default': c1.selectbox('Credit in default', options['default']),
                'housing': c2.selectbox('Housing loan', options['housing'], index=1),
                'loan': c3.selectbox('Personal loan', options['loan']),
                'contact': c4.selectbox('Contact type', options['contact']),
                'month': c1.selectbox('Month', options['month'], index=options['month'].index('may')),
                'day_of_week': c2.selectbox('Day of week', options['day_of_week']),
                'campaign': c3.number_input('Calls this campaign (incl. this one)', 1, 60, 1),
                'previous': c4.number_input('Contacts before this campaign', 0, 10, 0),
                'poutcome': c1.selectbox('Previous campaign outcome', options['poutcome'],
                                         index=options['poutcome'].index('nonexistent')),
            }
            days = c2.number_input('Days since last contact (blank if never)', 0, 998, 0)
            never = c3.checkbox('Never contacted before', value=True)
            submitted = st.form_submit_button('Score customer', type='primary')
        # Results are kept in the session, so using one control doesn't clear the other's result.
        if submitted:
            record['pdays'] = 999 if never else int(days)
            try:
                st.session_state['single'] = scoring.score(runtime, endpoint_name, [record], schema)[0]
            except Exception as error:
                st.error(f'Scoring failed: {error}')
        if 'single' in st.session_state:
            probability = st.session_state['single']
            c1, c2 = st.columns(2)
            c1.metric('Probability of subscribing', f'{probability:.1%}')
            c2.metric('Decision', 'Call first' if probability >= threshold else 'Lower priority',
                      delta=f'{probability - threshold:+.1%} vs threshold')
            st.progress(min(probability, 1.0))

        st.subheader('Current results: the newest customers')
        n = st.slider('How many of the newest customers to score', 20, 500, 200, step=20)
        if st.button('Score the newest customers'):
            customers = recent_customers(config['data']['path'], config['data']['separator'], n)
            features_only = customers.drop(columns=['y', 'duration'])
            try:
                scores_ = scoring.score(runtime, endpoint_name, features_only.to_dict('records'), schema)
            except Exception as error:
                st.error(f'Scoring failed: {error}')
            else:
                st.session_state['newest'] = scoring.rank(
                    features_only[['age', 'job', 'contact', 'month', 'campaign', 'poutcome']], scores_, threshold
                ).assign(subscribed=customers['y'].eq('yes'))
        if 'newest' in st.session_state:
            ranked = st.session_state['newest']
            truth = ranked['subscribed']
            flagged = ranked[ranked['call_first']]
            c1, c2, c3 = st.columns(3)
            c1.metric('Flagged "call first"', f'{len(flagged)} of {len(ranked)}')
            c2.metric('Subscribed among flagged', f'{flagged["subscribed"].mean():.0%}' if len(flagged) else '-')
            c3.metric('Subscribed overall', f'{truth.mean():.0%}')
            st.altair_chart(alt.Chart(ranked, title='Score distribution').mark_bar().encode(
                x=alt.X('score:Q', bin=alt.Bin(maxbins=30)), y='count()',
                color=alt.Color('subscribed:N', title='actually subscribed')))
            st.dataframe(ranked.head(50), hide_index=True)
            st.caption('These customers are historical, so their real outcome is shown for comparison. Every '
                       'request is also recorded by Data Capture for monitoring.')
