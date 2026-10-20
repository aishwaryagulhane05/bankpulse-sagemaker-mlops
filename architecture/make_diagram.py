"""Draw architecture/architecture.png: the local framework and the SageMaker components in the lab.

Run from anywhere:
    python architecture/make_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # no window needed; render straight to a file
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / 'architecture.png'

LOCAL, AWS, MONITOR = '#E8F1FB', '#FFF4E5', '#EAF7EE'
EDGE = {LOCAL: '#2F6DB5', AWS: '#E07B00', MONITOR: '#2E8B57'}

# name: (x, y, width, height, title, detail, colour); coordinates in a 0-100 x 0-60 canvas.
BOXES = {
    'raw':      (2, 46, 14, 9, 'Raw data', 'bank-additional-full.csv\n41,188 calls, 2008-2010', LOCAL),
    'contract': (19, 46, 14, 9, 'Data contract', 'schema + 4 rules\ndrop duration (leakage)', LOCAL),
    'split':    (36, 46, 14, 9, 'Time split', '70 / 15 / 15\noldest rows train', LOCAL),
    'features': (53, 46, 16, 9, 'Feature engine', '12 versioned features\nfitted on train only', LOCAL),
    'outputs':  (72, 46, 26, 9, 'Pipeline outputs', 'XGBoost CSVs, feature_schema.json,\nrun_metadata.json (hashes, git commit)', LOCAL),

    's3':       (72, 29, 26, 10, 'Amazon S3 data lake', 'raw/  train/  validation/  batch-input/\nlabels/  feature-schema/  model-artifacts/', AWS),
    'train':    (50, 29, 19, 10, 'SageMaker Training Job', 'built-in XGBoost\nml.m5.large -> model.tar.gz', AWS),
    'batch':    (28, 29, 19, 10, 'Batch Transform', 'score validation + test', AWS),
    'eval':     (2, 29, 23, 10, 'Evaluation', 'threshold on validation, test once\nPR-AUC vs logistic baseline', LOCAL),

    'registry': (2, 12, 23, 10, 'Model Registry + Model Card', 'version -> Approved\nintended use, limitations', AWS),
    'endpoint': (28, 12, 19, 10, 'Real-time Endpoint', 'encode_records()\n-> InvokeEndpoint', AWS),
    'capture':  (50, 12, 19, 10, 'Data Capture', 'requests + scores\n-> inference-capture/', AWS),
    'monitor':  (72, 12, 26, 10, 'Monitoring', 'PSI drift, prediction drift,\nage-band selection rates', MONITOR),
    'cloudwatch': (72, 1, 26, 8, 'Amazon CloudWatch', 'custom metrics + alarm', MONITOR),
}

# (from, to, from side, to side)
ARROWS = [
    ('raw', 'contract', 'right', 'left'), ('contract', 'split', 'right', 'left'),
    ('split', 'features', 'right', 'left'), ('features', 'outputs', 'right', 'left'),
    ('outputs', 's3', 'bottom', 'top'), ('s3', 'train', 'left', 'right'),
    ('train', 'batch', 'left', 'right'), ('batch', 'eval', 'left', 'right'),
    ('eval', 'registry', 'bottom', 'top'), ('registry', 'endpoint', 'right', 'left'),
    ('endpoint', 'capture', 'right', 'left'), ('capture', 'monitor', 'right', 'left'),
    ('monitor', 'cloudwatch', 'bottom', 'top'),
]


def anchor(name, side):
    x, y, w, h = BOXES[name][:4]
    return {'left': (x, y + h / 2), 'right': (x + w, y + h / 2),
            'top': (x + w / 2, y + h), 'bottom': (x + w / 2, y)}[side]


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9.6), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis('off')

    # Step 1: one rounded box per component, title in bold and details below.
    for x, y, w, h, title, detail, colour in BOXES.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1',
                                    facecolor=colour, edgecolor=EDGE[colour], linewidth=1.6))
        ax.text(x + w / 2, y + h * 0.70, title, ha='center', va='center', fontsize=10.5, fontweight='bold')
        ax.text(x + w / 2, y + h * 0.32, detail, ha='center', va='center', fontsize=8.3, color='#333333')

    # Step 2: arrows along the data flow.
    for start, end, start_side, end_side in ARROWS:
        ax.add_patch(FancyArrowPatch(anchor(start, start_side), anchor(end, end_side), arrowstyle='-|>',
                                     mutation_scale=14, linewidth=1.4, color='#555555',
                                     shrinkA=4, shrinkB=4))

    # Step 3: title and legend for where each part runs.
    ax.text(50, 60, 'bankpulse: term deposit subscription model on AWS SageMaker', ha='center',
            fontsize=15, fontweight='bold')
    legend = [(LOCAL, 'Local Python framework (bankpulse_core + features), tested with pytest'),
              (AWS, 'AWS Academy lab: SageMaker components, run as LabRole'),
              (MONITOR, 'Monitoring: drift and subgroup report, CloudWatch metrics')]
    for i, (colour, label) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((2, 6.2 - i * 2.6), 2.2, 1.6, boxstyle='round,pad=0.1',
                                    facecolor=colour, edgecolor=EDGE[colour]))
        ax.text(5.2, 7.0 - i * 2.6, label, va='center', fontsize=8.8)

    fig.savefig(OUTPUT, bbox_inches='tight', facecolor='white')
    print(f'saved {OUTPUT}')


if __name__ == '__main__':
    main()
