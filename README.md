# Lending Club Loan Default Prediction Model

## Project Overview

This project builds a machine learning model to predict loan default probability using historical Lending Club data. The model calculates portfolio-level Point-in-Time Probability of Default (PD) with confidence intervals to assess credit risk.

---

## Table of Contents

- [Data Source](#data-source)
- [How to Download the Data](#how-to-download-the-data)
- [Model Overview](#model-overview)
- [What the Model Calculates](#what-the-model-calculates)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results Interpretation](#results-interpretation)

---

## Data Source

**Dataset**: Lending Club Loan Data (2007-2018 Q4)

**Kaggle Link**: https://www.kaggle.com/datasets/wordsforthewise/lending-club

The dataset contains comprehensive information about loans issued through Lending Club, including borrower characteristics, loan details, credit history, and loan outcomes.

---

## How to Download the Data

### Option 1: Using Kaggle Website

1. **Create a Kaggle Account**
   - Go to https://www.kaggle.com and create a free account if you don't have one
   - Verify your email address

2. **Navigate to the Dataset**
   - Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club
   - Click the "Download" button (requires login)

3. **Extract the Data**
   - Unzip the downloaded file
   - Locate the file: `accepted_2007_to_2018Q4.csv`
   - Place it in your project directory

### Option 2: Using Kaggle API (Recommended for Automation)

1. **Install Kaggle API**
   ```bash
   pip install kaggle
   ```

2. **Set Up API Credentials**
   - Go to your Kaggle Account settings: https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json` with your credentials
   - Place it in:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<username>\.kaggle\kaggle.json`

3. **Download the Dataset**
   ```bash
   kaggle datasets download -d wordsforthewise/lending-club
   ```

4. **Unzip the Files**
   ```bash
   unzip lending-club.zip
   ```

5. **Verify the File**
   - Ensure `accepted_2007_to_2018Q4.csv` is in your working directory

---

## Model Overview

### Model Type
**LightGBM Classifier** - A gradient boosting framework optimized for:
- High performance on large datasets
- Handling imbalanced data (loan defaults are rare events)
- Fast training speed
- Low memory usage

### Prediction Target
The model predicts **loan default** as a binary classification:
- **Class 0**: Loan will be fully paid (no default)
- **Class 1**: Loan will default (charged off, late payments, etc.)

### Training Approach
1. **Data Preprocessing**: Cleaning, handling missing values, feature engineering
2. **Feature Encoding**: Converting categorical variables (purpose, home ownership, etc.) to numerical format
3. **Train-Test Split**: Splitting data to evaluate model performance on unseen data
4. **Model Training**: Training LightGBM on historical loan outcomes
5. **Prediction**: Generating default probabilities for each loan

---

## What the Model Calculates

### 1. Individual Loan Default Probability
For each loan, the model outputs:
- **Default Probability** (0 to 1): Likelihood that the loan will default
- Example: 0.2194 = 21.94% chance of default

### 2. Portfolio-Level Point-in-Time PD
The model aggregates individual predictions to calculate:

**Portfolio PD** = Average default probability across all loans in the portfolio

This represents the expected default rate for the entire loan portfolio at the current point in time.

### 3. Confidence Intervals
The model provides two types of uncertainty estimates:

#### Bootstrap Confidence Interval (Sampling Uncertainty)
- Based on 1,000 bootstrap resamples
- Captures variability in portfolio composition
- Narrower interval reflecting sampling variation

#### Recall-Adjusted Confidence Interval (Model Uncertainty)
- Accounts for model's ability to detect actual defaults
- Formula: `Margin of Error = Portfolio_PD × (1 - Recall)`
- Wider interval reflecting model imperfection
- **More conservative and realistic** for risk management

**Example Output:**
```
Portfolio PD: 21.94%

Bootstrap CI (sampling uncertainty):
  95% CI: [21.85%, 22.03%]

Recall-Adjusted CI (model uncertainty):
  95% CI: [15.96%, 27.92%]
  Margin of Error: ±5.98%
```

---

## Model Performance

Based on test data evaluation:

| Metric | Value | Meaning |
|--------|-------|---------|
| **Recall** | 72.74% | Model correctly identifies 72.74% of actual defaults |
| **Precision** | 81.62% | When model predicts default, it's correct 81.62% of the time |
| **F1-Score** | 76.92% | Harmonic mean of precision and recall |

### What This Means:
- **High Recall**: Model catches most defaults (good for risk management)
- **High Precision**: Low false alarm rate
- **Trade-off**: Some defaults are missed (~27% false negatives)

---

## Key Features

The model uses multiple features from the Lending Club dataset, including:

### Loan Characteristics
- Loan amount
- Interest rate
- Loan term (36 or 60 months)
- Installment amount
- Loan grade and sub-grade
- Loan purpose (debt consolidation, credit card, home improvement, etc.)

### Borrower Information
- Annual income
- Employment length
- Home ownership status
- Debt-to-income ratio (DTI)
- Verification status

### Credit History
- FICO score range
- Number of delinquencies in last 2 years
- Earliest credit line date
- Number of inquiries in last 6 months
- Revolving balance and utilization
- Total number of accounts
- Public records (bankruptcies, tax liens)

All categorical features (like loan purpose) are one-hot encoded into binary columns.

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install pandas numpy scikit-learn lightgbm
```

### Project Structure
```
lending-club-prediction/
│
├── accepted_2007_to_2018Q4.csv    # Downloaded data
├── code.ipynb                      # Main notebook
├── README.md                       # This file
└── requirements.txt                # Dependencies
```

### Create requirements.txt
```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Load the Data
```python
import pandas as pd
df = pd.read_csv('accepted_2007_to_2018Q4.csv', low_memory=False)
```

### Step 2: Train the Model
Run the preprocessing and training cells in the notebook to:
- Clean and prepare the data
- Split into training and test sets
- Train the LightGBM model

### Step 3: Generate Predictions
```python
# Get default probabilities
default_probability = trained_models['LightGBM'].predict_proba(X_test)[:, 1]

# Get binary predictions
y_pred = trained_models['LightGBM'].predict(X_test)
```

### Step 4: Calculate Portfolio PD with Confidence Intervals
```python
from sklearn.utils import resample
import numpy as np

# Bootstrap to calculate confidence intervals
portfolio_pds = []
for i in range(1000):
    boot_indices = resample(range(len(default_probability)), random_state=i)
    boot_pds = default_probability[boot_indices]
    portfolio_pds.append(boot_pds.mean())

# Calculate metrics
portfolio_pd = np.mean(portfolio_pds)
ci_lower_bootstrap = np.percentile(portfolio_pds, 2.5)
ci_upper_bootstrap = np.percentile(portfolio_pds, 97.5)

# Adjust for model recall
recall = recall_score(y_test, y_pred)
margin_of_error = portfolio_pd * (1 - recall)
ci_lower_adjusted = max(0, portfolio_pd - margin_of_error)
ci_upper_adjusted = min(1, portfolio_pd + margin_of_error)
```

---

## Results Interpretation

### Portfolio PD: 21.94%
- Expected that approximately 22% of loans in the portfolio will default
- This is the point estimate based on model predictions

### Bootstrap CI: [21.85%, 22.03%]
- Very narrow range due to large sample size
- Reflects sampling variability only
- **Not recommended for risk decisions** (too optimistic)

### Recall-Adjusted CI: [15.96%, 27.92%]
- **Recommended for risk management**
- Accounts for the fact that model misses ~27% of defaults
- True default rate could be as low as 16% or as high as 28%
- Margin of error: ±5.98%

### Practical Application

**For Risk Management:**
- Conservative estimate: Plan for up to 27.92% default rate
- Expected estimate: Use 21.94% for baseline scenarios
- Optimistic estimate: 15.96% represents best-case scenario

**For Portfolio Decisions:**
- If 28% default rate is unacceptable → Tighten lending criteria
- If 16% default rate is manageable → Current strategy is viable
- Monitor actual defaults to calibrate model over time

---

## Model Limitations

1. **Temporal Validity**: Model trained on 2007-2018 data; economic conditions change
2. **Recall Constraint**: Misses 27% of actual defaults (false negatives)
3. **Class Imbalance**: Defaults are rare events, affecting model calibration
4. **Feature Availability**: Requires all input features to be available for new predictions
5. **No Causal Inference**: Model identifies correlations, not causes of default

---

## Future Improvements

- Implement SMOTE or other techniques for handling class imbalance
- Feature engineering to capture non-linear relationships
- Time-series validation to account for temporal patterns
- Ensemble with multiple models for improved robustness
- Regular retraining on fresh data to maintain accuracy
- Explainability analysis (SHAP values) to understand key risk drivers

---

## License

This project uses publicly available data from Lending Club via Kaggle. Please refer to Kaggle's terms of use and the original data source for licensing information.

---

## Contact & Contributions

For questions, suggestions, or contributions, please open an issue or submit a pull request.

---

## Acknowledgments

- **Lending Club** for making the data publicly available
- **Kaggle** for hosting the dataset
- **LightGBM** developers for the excellent gradient boosting library
