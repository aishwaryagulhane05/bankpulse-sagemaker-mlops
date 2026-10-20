"""Text for the overview tab: what the project is, why it matters and who it is for."""

TITLE = 'BankPulse: who to call first'

TAGLINE = ('A SageMaker model that scores each bank customer\'s chance of opening a term deposit **before** the '
           'call, so a campaign team can call the most promising customers first.')

PROBLEM = '''
A bank sells term deposits by phone, and only about **1 in 9 calls** ends in a subscription. Agents work through call
lists in no particular order, so most of their time goes on customers who were never going to say yes, and those
customers get calls they didn't want.

Call length is known only after the call, when the outcome is already known, so it can't be used. The model may only use
what the bank knows **before** dialling: the customer profile, the contact history and the campaign so far.
'''

WHY_IT_MATTERS = '''
- **Agent time:** calling the top-scored customers first finds more subscribers in the same number of calls.
- **Customer experience:** fewer unwanted calls to people unlikely to be interested.
- **Responsible use:** the model is versioned, approved and documented (Model Card), and monitored for drift and for
  differences between age groups, so problems are seen instead of silently affecting who gets called.
- **Changing conditions:** subscription rates moved from 6% to 38% between 2008 and 2010. Monitoring and planned
  retraining keep the model current instead of frozen in one period.
'''

AUDIENCE = [
    ('Campaign managers', 'Plan call lists; decide how many customers to call in a campaign.',
     'A ranked list and the expected share of subscribers at each cut-off.'),
    ('Call-centre agents', 'Make the calls.', 'Who to call next, highest score first.'),
    ('Data science / MLOps team', 'Build, deploy and watch the model.',
     'Training, registry, endpoint and drift status in one place.'),
    ('Risk and compliance', 'Check the model is used appropriately.',
     'Model Card, approval decisions and selection rates by age band.'),
]

HOW_IT_HELPS = '''
1. The campaign team uploads or selects the customers for the next campaign.
2. Each customer is scored by the SageMaker endpoint, a probability of subscribing.
3. Customers above the threshold (chosen on validation data) are flagged **call first**, and everyone is ranked by score.
4. Agents call down the list. On the newest test period, the threshold finds **35% more subscribers** than the
   logistic baseline (718 vs 533).
'''

NOT_FOR = ('Not for credit, lending, pricing or any decision about who may get a product. It ranks marketing calls '
           'only, and campaign staff review the list.')

LIFECYCLE = [
    ('Data lake', 'S3', 'Raw data, time-ordered splits, feature schema, lineage'),
    ('Training', 'SageMaker Training Job', 'Built-in XGBoost on ml.m5.large'),
    ('Batch scoring', 'SageMaker Batch Transform', 'Validation and test predictions'),
    ('Governance', 'Model Registry + Model Card', 'Versioned, approved, documented'),
    ('Serving', 'Real-time endpoint + Data Capture', 'Live scores, every request logged'),
    ('Monitoring', 'CloudWatch', 'Drift and age-band selection metrics, alarm, dashboard'),
]
