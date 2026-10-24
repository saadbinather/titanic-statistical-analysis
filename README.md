# Titanic Survival Detector — EDA

An executable, statistically grounded investigation of survival patterns in the Titanic passenger manifest. The repository preserves the original `model.py` demonstration while adding a reproducible notebook-based research workflow.

## Scope

The notebooks investigate:

- data provenance, schema quality, duplicates, target imbalance, and missingness;
- missing-data patterns and evidence relevant to the MCAR assumption;
- univariate, bivariate, and multivariate survival structure;
- Pearson/Spearman associations and bias-corrected Cramér's V;
- confidence intervals, bootstrap uncertainty, chi-square tests, Mann–Whitney tests, effect sizes, and multiple-comparison caution;
- interactions between sex and passenger class, outliers, skew, and confounding;
- leakage-safe preprocessing and stratified cross-validated Logistic Regression and Random Forest models;
- held-out diagnostics, permutation importance, feature engineering, and PCA as a descriptive dimensionality-reduction lens.

The central discipline is interpretive: associations are not presented as causal effects, and predictive importance is not treated as a mechanism.

##  Structure

```text
.
├── titanic.csv
├── model.py                         # Original end-to-end analysis script
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_inference_and_hypothesis_tests.ipynb
│   └── 03_interpretable_modeling_and_pca.ipynb
├── requirements.txt
└── README.md
```

## Reproducible setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numerical order. Each notebook locates `titanic.csv` from the repository root when launched from `notebooks/` or the root directory. Random seeds are fixed where stochastic procedures are used.

## Notebook narrative

### 1. Data audit and EDA

Establishes what is observed before modeling. It quantifies missingness, visualizes distributions and interactions, and computes numeric and categorical association measures. The notebook explicitly distinguishes marginal patterns from stratified patterns that may reflect confounding.

### 2. Inference and hypothesis tests

Turns prominent visual patterns into explicit null hypotheses. It reports effect sizes and bootstrap confidence intervals alongside p-values. Tests are framed as evidence under assumptions—not proof of a causal story.

### 3. Interpretable modeling and PCA

Estimates out-of-sample predictive performance with preprocessing contained inside cross-validation folds. It reports several metrics, held-out diagnostics, permutation importance, family-size engineering, and PCA visualization. Explanations are model-conditional and not causal.

## Key analytical safeguards

- `Survived` is never used in preprocessing or feature construction.
- Imputation, scaling, and one-hot encoding are fitted within each training fold.
- Class imbalance is addressed through metrics and model configuration rather than accuracy alone.
- Fare is treated as skewed; robust rank-based tests supplement visual summaries.
- Correlated predictors are not assigned independent causal interpretations.
- Historical, observational data limits transportability and causal claims.

## Dataset

The included CSV contains `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`. `Survived` is binary: `0` means did not survive and `1` means survived.

## Original script

The original workflow remains available:

```bash
python model.py
```

It produces introductory plots and compares Logistic Regression with Random Forest. The notebooks are the recommended research-grade workflow.

## Limitations

This is an observational historical dataset with recording artifacts, missingness, selection effects, and limited population coverage. Results describe this sample and should not be interpreted as universal claims about survival, fairness, or modern emergency response.

## Extended statistical research layer

The repository also includes a dual-language research pipeline:

- `scripts/research_pipeline.py` implements sigmoid/BCE logistic regression with NumPy gradient descent, compares held-out Logistic Regression and Random Forest models, computes ROC-AUC/PR-AUC/Brier scores, creates calibration and learning curves, and saves 300-DPI figures.
- `R/statistical_analysis.R` fits multivariable logistic regression with interaction effects, odds-ratio intervals, VIF, likelihood-ratio comparison, AIC/BIC, residual diagnostics, and saved ggplot2 figures.
- `R/statistical_analysis.Rmd` provides the literate R version with descriptive statistics, calibration, diagnostics, and bootstrap uncertainty.
- `renv/README.md` documents how to create a machine-specific R lockfile when R is available.

Run the Python research pipeline from the repository root:

```bash
python scripts/research_pipeline.py
```

Run the R analysis from the repository root:

```bash
Rscript R/statistical_analysis.R
```

Generated artifacts are written to `figures/eda/`, `figures/statistics/`, `figures/modeling/`, `figures/bias_variance/`, and `results/`. The current execution environment did not include R or the Python scientific stack, so no unverified numerical results are included in this README.

## License

Educational and experimental use.
