# MultiTask-Battery-Prediction

Code for paper "Co-Estimation of SOH and RUL For Second-Life Battery Based on Self-Supervised Auxiliary Learning"

A multi-task learning framework for battery health prognostics with gradient reversal technique, capable of simultaneously predicting State of Health (SOH), Remaining Useful Life (RUL), and Discharge Capacity Change Rate (DCCR).

### Usage
1. Hyperparameter Optimization
Run the hyperparameter optimization notebook first to find optimal parameters:
jupyter notebook HyperOpt.ipynb

3. Model Training and Testing
After obtaining optimal hyperparameters.
jupyter notebook Co-estimation_code.ipynb

### Model Architecture
The framework consists of:

Shared Encoder: Extracts common features from input battery data

Task-specific Predictors:

SOH Predictor: Estimates SOH

RUL Predictor: Predicts RUL

DCCR Predictor: Estimates DCCR

Auxiliary Network: Promotes feature disentanglement via gradient reversal

### Uncertainty Estimation
Usage Example
python
#### Evaluation without uncertainty
rmse, mae, mape, r2, predictions = model_test(
    model, test_loader, test_targets, uncertainty_state=False
)

#### Evaluation with uncertainty
rmse, mae, mape, r2, predictions, std = model_test(
    model, test_loader, test_targets, uncertainty_state=True
)
