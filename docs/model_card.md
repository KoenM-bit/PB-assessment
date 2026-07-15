# Model Card

Each training run that completes MLflow logging produces a `model_card.json` artifact
under `reports/` (and locally at `artifacts/model/model_card.json`).

## Contents

| Section | Description |
|---------|-------------|
| `model_description` | Model type, intended use, limitations |
| `training_data` | Row counts, Delta table versions, git commit |
| `features` | Feature pipeline version and feature list |
| `metrics` | Holdout, walk-forward, and segment MAE |
| `quality_gates` | Pass/fail status and threshold details |
| `experimentation` | Optional tuning, ablation, SHAP summaries |
| `ethical_considerations` | Segment fairness notes |
| `lineage` | MLflow run ID and config paths |

## Where to find it

- **MLflow UI**: open a training run → Artifacts → `reports/model_card.json`
- **Local**: `make train` writes `artifacts/model/model_card.json`
- **Promotion audit**: compare `gate_report.json` and `model_card.json` before `make promote-challenger`

## Regenerating

Model cards are generated automatically in `train()` — do not hand-edit the JSON.
Adjust thresholds in `ml/config/quality_gates.yaml` or training settings in
`ml/config/training.yaml`, then re-run training.
