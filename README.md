# Ensemble Health — Claims Denial Prediction

A machine learning pipeline for evaluating and predicting health insurance claim denials, featuring false negative analysis, feature engineering insights, and SHAP-based feature importance.

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
