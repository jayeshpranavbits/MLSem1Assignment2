# Dry Bean Classification (6 Models)

## Problem Statement
The objective of this project is to classify dry beans into one of seven categories
based on geometric and morphological features using supervised machine learning techniques.

## Dataset
Dry Bean Dataset (public repository).  
- Instances: 13,611  
- Features: 16 numerical attributes
- Classes: 7 bean types (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)

## Models Implemented
1. Logistic Regression (Multinomial)  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest  
6. XGBoost  

## Evaluation Metrics 
- Accuracy  
- AUC (multi-class ROC-AUC, OvR macro)  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  
- MCC (Matthews Correlation Coefficient)  

## Deployment:
The application is deployed using Streamlit Community Cloud and allows users to:
- Upload a CSV dataset
- Select one of six ML models
- View evaluation metrics
- View confusion matrix
- Download predictions (if labels are not provided)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Observations:
XGBoost achieved the highest macro F1-score and overall accuracy, demonstrating
superior performance in handling multi-class classification tasks.
ML Model Name	      | Accuracy	Precision	Recall	  F1	      MCC	       AUC
Logistic Regression	| 0.921410	0.935383	0.932149	0.933538	0.905045	0.994776
Decision Tree       |	0.892031	0.907513	0.909028	0.908061	0.869569	0.944996
kNN		              | 0.916269  0.931763	0.926738	0.928868	0.898792	0.986807
Naives-Bayes	      | 0.763863	0.774427	0.769417	0.767750	0.715406	0.967193
Random Forest	      | 0.920308	0.934654	0.930010	0.932210	0.903591	0.993567
XGBoost		          | 0.925450  0.939923	0.935143	0.937430	0.909807	0.995291


Observations:
ML Model Name	 \\\\ Observation on Model Performance and Output
Logistic Regression	- As a linear classifier with multinomial optimization, Logistic Regression achieved strong macro-averaged metrics, indicating effective separation of classes in feature space. However, its performance slightly declined for structurally similar beans, highlighting limitations in modelling nonlinear feature interactions.
Decision Tree -	The Decision Tree captured nonlinear decision boundaries but exhibited higher variance compared to ensemble methods. While it performed reasonably well, slight inconsistencies across classes suggest sensitivity to training data splits and potential overfitting.
kNN	- kNN demonstrated competitive macro F1 performance by leveraging distance-based classification. However, its effectiveness depended heavily on feature scaling and class density, and performance degraded slightly in regions with overlapping morphological features.
Naive Bayes	Gaussian - Naive Bayes produced moderate results due to its strong independence assumption among features. Since geometric attributes in the dataset are correlated, this assumption reduced its ability to model complex class boundaries accurately.
Random Forest (Ensemble) - Random Forest improved predictive stability by aggregating multiple decision trees, significantly reducing variance and improving macro-level metrics. It handled nonlinear relationships effectively and showed better class balance compared to single-tree models.
XGBoost (Ensemble) - XGBoost achieved the highest macro F1-score and overall accuracy due to its gradient boosting framework, which sequentially corrected previous errors. Its regularization and optimized tree-building strategy allowed superior handling of subtle inter-class differences, especially among morphologically overlapping bean types.


## Notes for files placed in project
- The app expects a CSV containing the same feature columns plus a target column named `Class` (preferred) or as the last column.
- Trained models and label encoder are stored in the `model/` folder.
