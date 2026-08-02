# Ensemble Health — Claims Denial Prediction

A machine learning pipeline for evaluating and predicting health insurance claim denials, featuring false negative analysis, feature engineering insights, and SHAP-based feature importance.

## Table of Important Contents

- **Final Report:** [Predicting Claim Denials.pdf](https://github.com/shivAM2886/ensemble-claim-denial/blob/main/Predicting%20Claim%20Denials.pdf)
- **Predictions on Current Claims:** [current_claims_with_denial_probs.csv](https://github.com/shivAM2886/ensemble-claim-denial/blob/main/data/current_claims_with_denial_probs.csv)
- **Top 10 Riskiest Current Claims (with LLM Explanations):** [current_claims_top_10_risk_with_denial_probs_explanations.csv](https://github.com/shivAM2886/ensemble-claim-denial/blob/main/data/current_claims_top_10_risk_with_denial_probs_explanations.csv)


## Requirements

- **Python 3.11 or higher**
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment management)

## Setup

### 1. Create a Conda environment

```bash
conda create -n ensemble-health python==3.11
conda activate ensemble-health
```

> **Note:** This project requires Python 3.11 at minimum to run.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run evaluation and prediction

To run the evaluation and prediction using the existing trained model:

```bash
python main.py
```

This will:

- Read the historical claims data from the `data/` folder
- Run the model evaluation
- Write the **false negative analysis**
- Generate **predictions** on the current `claims.csv` file
- Produce **feature engineering denial rate tables**
- Generate a **SHAP feature importance** plot (`.png`)

### Retrain the model

To retrain the model before running evaluation and prediction, use the `--train` flag:

```bash
python main.py --train
```

This performs the same steps as above, but retrains the model first.

## Outputs

| Output | Description |
| --- | --- |
| False negative analysis | Analysis of denials ranked outside of top 25% |
| Predictions | Denial predictions on the current `current_claims.csv` |
| Denial rate tables | Feature engineering denial rate breakdowns |
| SHAP importance plot | `.png` visualizing feature importance |
