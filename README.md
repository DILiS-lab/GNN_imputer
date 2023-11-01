# Protein Abundance Imputation with Graph Neural Networks

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

This repository contains the code and resources for a protein abundance imputation framework using a graph neural network approach. The framework is designed to predict missing protein abundances by leveraging relationships between gene-derived molecules such as proteins, peptides, and mRNAs. It offers a flexible and high-performing solution for imputing protein abundance values in mass spectrometry-driven proteomics data.

### Features

- Inductive, attention-based graph neural network framework
- Ability to utilize all available abundances for protein-peptide-mRNA tuples
- Flexible graph-building and model training scheme
- Benchmarking against different imputation methods on diverse human datasets

## Installation

To use this framework, you'll need to set up a Python environment. We recommend using a virtual environment to manage your dependencies. Here are the basic installation steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/DILiS-lab/GNN_imputer.git

2. Navigate to the project directory:
   ```bash
    cd GNN_imputer

3. Install the required packages:
   ```bash
    pip install -r requirements.txt

## Usage

To run the GNN model, follow these steps:
1. Navigate to the "train" folder:
    ```bash
    cd train
2. Run the GNN model with the following command:
   ```python
    python3 train_gnn_model.py <FOLD_NUM>

The FOLD_NUM is the fold that you wish to train on (Should be between 0 to 4). For running training for all folds use 'all' for FOLD_NUM.
The 'data' folder contains the DGL networks for the Blood Plasma dataset on which the GNN is to be trained.

## Results
The training generates results in the form of:
  - The foldwise performance over epochs in the file _train/performance_GNN_regression.csv_. 
  - The pytorch saved model in the _train/saved_models_ directory with file name as best_model_fold_<FOLD_NUM>.pt 
  - The outputs predictions for the known test molecules in the _train/saved_models_ directory with file name as mapping_abundances_prediction_vs_original_fold_<FOLD_NUM>.json.
  - The outputs predictions for the unknown test molecules in the _train/saved_models_ directory with file name as mapping_abundances_prediction_vs_missing_original_fold_<FOLD_NUM>.json.

## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
