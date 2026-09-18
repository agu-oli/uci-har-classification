# UCI HAR – Machine Learning and Deep Learning Classification

This project analyzes the **UCI Human Activity Recognition (HAR) dataset** and compares two approaches for activity classification:

- Support Vector Machines (SVM), including PCA/LDA dimensionality reduction and hyperparameter tuning.
- A fully connected neural network implemented with TensorFlow/Keras.

The workflow includes exploratory data analysis, subject-aware train/validation splitting, 5-fold `StratifiedGroupKFold` cross-validation, model selection, and final evaluation on the test set.

## Files

- `exploratory_analysis.ipynb` – exploratory analysis, correlations, PCA and LDA.
- `svm.ipynb` – SVM baselines, GridSearchCV, cross-validation and test evaluation.
- `deep_learning_model.ipynb` – neural-network architectures, early stopping, cross-validation and test evaluation.
- `relazione.pdf` – project report.

## Setup

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Then open the notebooks with Jupyter and run them in order.
