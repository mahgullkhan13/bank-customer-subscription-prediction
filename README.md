# Bank Customer Subscription Prediction

A machine learning classification project that predicts whether a bank customer will subscribe to a term deposit. The project compares a custom k-Nearest Neighbours (kNN) classifier implemented from scratch in Python with a Scikit-learn Decision Tree classifier.

## Project Overview

The objective of this project was to investigate and compare two different machine learning approaches for predicting customer responses to bank marketing campaigns.

The project uses the Bank Marketing dataset containing customer information and outcomes from telephone-based marketing campaigns.

A key part of the project was implementing the k-Nearest Neighbours algorithm from scratch rather than relying on Scikit-learn's built-in kNN classifier.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

## Machine Learning Models

### k-Nearest Neighbours (kNN)

The kNN classifier was implemented from scratch using Python and NumPy.

The implementation includes:

- Vectorised Euclidean distance calculation
- Identification of nearest neighbours using NumPy
- Majority-vote classification
- Experimentation with different values of `k`

Values of `k` including 3, 5, 7 and 11 were evaluated.

### Decision Tree

A Decision Tree classifier was implemented using Scikit-learn and compared against the custom kNN model.

Different maximum tree depths were evaluated to investigate model complexity, underfitting and overfitting.

## Data Preprocessing

The workflow included:

- Converting the target variable (`deposit`) from yes/no to binary values
- Selecting numerical features
- Performing an 80/20 stratified train-test split
- Standardising numerical features for the kNN classifier
- Maintaining an unscaled feature set for the Decision Tree

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- 5-fold stratified cross-validation
- Hyperparameter experimentation

## Results

The baseline models produced approximately:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| kNN (k=7, from scratch) | 77.5% | 75.8% | 77.2% | 76.5% |
| Decision Tree (depth=5) | 78.9% | 80.5% | 73.2% | 76.7% |

Further experimentation showed that kNN performance improved to approximately **78.1% accuracy with k=11**, while Decision Tree configurations around depths 4–6 achieved approximately **78.7–78.9% accuracy**.

Overall, the Decision Tree achieved slightly higher accuracy and precision, while the custom kNN classifier achieved higher recall and comparable F1 performance.

## Key Learning Outcomes

This project strengthened my understanding of:

- Implementing machine learning algorithms from first principles
- Distance-based classification
- Feature scaling and preprocessing
- Train/test splitting and stratification
- Cross-validation
- Hyperparameter tuning
- Bias-variance trade-offs
- Model evaluation and comparison
- Overfitting and model complexity

## Repository Contents

- `bank_subscription_prediction.ipynb` — Complete notebook containing the implementation, experiments, evaluation and analysis
