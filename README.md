# Detection-of-Bankruptcy-
Machine learning project CSCI 3052U group 22

# Dataset Information:
[Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),[DOI](10.24432/C5F600)


This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license.This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given. 

## Data Card

### Dataset overview

- **Name:** Polish Companies Bankruptcy Data Set
- **Domain:** Business and finance
- **Purpose:** Predict whether a Polish company will go bankrupt from financial-ratio data.
- **Size:** 43,405 financial-statement records across five source files.
- **Inputs:** 64 numerical financial-ratio features.
- **Output:** `bankrupt`, where `0` means non-bankrupt and `1` means bankrupt.
- **Prediction horizons:** The files predict bankruptcy from five years ahead (`1year.arff`) to one year ahead (`5year.arff`).

### Provenance and collection

The data was collected from the Emerging Markets Information Service (EMIS) and distributed through the UCI Machine Learning Repository. UCI lists Sebastian Tomczak as the dataset creator. The source documentation does not specify ownership of the underlying EMIS records, funding sources, or the detailed sampling and label-annotation process.

Bankrupt companies were analyzed from 2000-2012, companies that are still operating were evaluated from 2007-2013.

### Preprocessing and transformations

The original ARFF files are unmodified and are in `data/raw/`.

For this project, exact duplicate rows were removed while keeping the first occurrence. Each source file was split separately into approximately 70% training, 15% validation, and 15% test data using stratified sampling with seed 42. Missing feature values were filled with medians calculated from the corresponding training split only. The target column and observed feature values were not changed.

No scaling, oversampling, undersampling, outlier removal, or feature engineering has been applied yet.

### Intended use

This dataset will be used for a machine learning project. The goal is to predict bankruptcy from financial ratios and compare appropriate classification models.

### Limitations and risks

- Bankruptcy cases are rare comparted to non-bankruptcy cases.
- Some records contain missing feature values.
- The dataset has no company identifier, thus company overlap across different horizons can not be analysed 
- Our findings may not generalize beyond Polish companies, but we will attempt to explore this further if time permits. 

**Source and license:** [UCI Machine Learning Repository - Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata), licensed under CC BY 4.0.
## Data folders
- `data/raw/`: original, unchanged ARFF datasets.
- `data/cleaned/`: cleaned training, validation and test files, plus supporting records.

Run notebooks from the repository root.
