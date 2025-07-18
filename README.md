# MultiTask-Battery-Prediction

MultiTask-Battery-Prediction

Code for paper "Co-Estimation of SOH and RUL For Second-Life Battery Based on Self-Supervised Auxiliary Learning"

A multi-task learning framework for battery health prognostics with gradient reversal technique, capable of simultaneously predicting State of Health (SOH), Remaining Useful Life (RUL), and Discharge Capacity Change Rate (DCCR).

🎯 Usage
1. Hyperparameter Optimization
Run the hyperparameter optimization notebook first to find optimal parameters:

2. Model Training and Testing
After obtaining optimal hyperparameters, run the main training script:


🏗 Model Architecture
The framework consists of:

Shared Encoder: Extracts common features from input battery data
Task-specific Predictors:
SOH Predictor: Estimates SOH
RUL Predictor: Predicts RUL
DCCR Predictor: Estimates DCCR
Auxiliary Network: Promotes feature disentanglement via gradient reversal
Input Battery Data
        ↓
   Shared Encoder
        ↓
   ┌─────────────────────────────────┐
   ↓                ↓                ↓
SOH Predictor   RUL Predictor   DCCR Predictor
   ↓                ↓                ↓
SOH Output      RUL Output      DCCR Output


📁 Files Description
Core Files
Co-estimation_code.ipynb: Main training and testing script
Model training with gradient reversal
Multi-task loss computation
Evaluation with/without uncertainty estimation

HyperOpt.ipynb: Hyperparameter optimization
Automated parameter search
Performance evaluation across parameter combinations
Best parameter selection

Key Components
Loss Functions:
MAPE Loss for SOH and RUL predictions
MSE Loss for DCCR reconstruction
Cross-entropy for auxiliary classifier

Optimizers:
Separate Adam optimizers for each component
Independent learning rates for fine-tuned training


🔗 Uncertainty Estimation
Usage Example
python
# Evaluation without uncertainty
rmse, mae, mape, r2, predictions = model_test(
    model, test_loader, test_targets, uncertainty_state=False
)

# Evaluation with uncertainty
rmse, mae, mape, r2, predictions, std = model_test(
    model, test_loader, test_targets, uncertainty_state=True
)
