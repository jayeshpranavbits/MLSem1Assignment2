# Dry Bean Classification (6 Models)

## Problem Statement
Classify dry beans into one of 7 categories using numerical shape features.

## Dataset
Dry Bean Dataset (public repository).  
- Instances: 13,611  
- Features: 16  
- Classes: 7 (bean types)

## Models Implemented
1. Logistic Regression (multinomial)  
2. Decision Tree  
3. KNN  
4. Gaussian Naive Bayes  
5. Random Forest  
6. XGBoost  

## Metrics Reported (per model)
- Accuracy  
- AUC (multi-class ROC-AUC, OvR macro)  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  
- MCC (Matthews Correlation Coefficient)  

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app expects a CSV containing the same feature columns plus a target column named `Class` (preferred) or as the last column.
- Trained models and label encoder are stored in the `model/` folder.
