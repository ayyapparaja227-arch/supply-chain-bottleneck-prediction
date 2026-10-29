# AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks

Stage 1 starter repository for data validation, EDA, baseline ML, evaluation, and bottleneck-risk classification.

## Stage 1 goal
Predict shipment delay risk and classify shipments as:
- `On-Time`
- `At-Risk`
- `Critical Delay`

The initial implementation is a baseline scaffold. Do not report model scores until the team runs evaluation on the selected dataset.

## Workflow
1. **Data intake** — collect shipment/delay data and supporting port/weather data.
2. **Data validation** — inspect schema, missingness, duplicates, target quality, and date/location coverage.
3. **EDA** — examine delay distribution and relationships with weather, waiting/dwell time, traffic, and capacity/utilization.
4. **Feature preparation** — define which features are available at prediction time; prevent target leakage.
5. **Baseline model** — use a time-aware or stratified split as appropriate, then train Logistic Regression and Random Forest baselines.
6. **Evaluation** — report precision, recall, F1, confusion matrix, and false-alarm rate. Agree on the acceptable false-alarm threshold with the team.
7. **Interpretation** — report feature importance and limitations.
8. **Risk output** — map model probabilities to the three risk categories only after thresholds are agreed and validated.
9. **Ops field guide** — create a one-page explanation of what each risk category means and what an operator should do.

## Repository layout
```text
data/
  raw/                 # Original datasets (normally do not commit large/private files)
  processed/           # Cleaned datasets (normally do not commit large files)
notebooks/
  01_eda.ipynb         # EDA notebook to create
src/
  data_validation.py
  train_baseline.py
  evaluate.py
  risk_rules.py
reports/
  figures/
  stage1_findings.md
tests/
requirements.txt
.gitignore
```

## Dataset guidance
Use one shipment-level dataset as the primary labelled dataset. Port dwell-time, port capacity/utilization, and historical weather sources can be supporting datasets. Do not merge them unless location and time keys are compatible and the join is defensible. If data are synthetic, label them as synthetic.

## Run locally
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/data_validation.py --input data/raw/your_shipment_file.csv
```

## Team collaboration
- Create issues for dataset selection, EDA, modelling, evaluation, and documentation.
- Use branches such as `feat/data-validation`, `feat/eda`, `feat/baseline-model`.
- Open pull requests instead of everyone pushing directly to `main`.
- Never commit API keys, credentials, or sensitive data.
