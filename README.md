# Titanic Survival Predictor

Predicts whether a passenger survived the Titanic disaster
using machine learning. Built with scikit-learn and a 
Random Forest classifier.

## Results
- Accuracy: 82.7%
- F1 Score: 76.3%
- Dataset: 891 passengersб 12 feautures

## What I learned
- How to handle missing data (median imputation for AGe)
- Feauture engineering (converting Sex, Embarked to numbers)
- Why accuracy alone is misleading on imbalanced datasets

## How to run
```bash
git clone https://github.com/doraduc/titanic-survival-ml
cd titanic-survival-ml
pip install -r requirements.txt
python titanic.py
```

## Tech stack
Python · scikit-learn · Pandas · NumPy · Random Forest

## Key insight
The most important feature for survival was Sex (women first 
policy) followed by passenger class — the model learned 
a 100-year-old historical social rule purely from numbers.

## Results
- Accuracy: 82.7% (local validation)
- F1 Score: 78.3%
- **Kaggle Public Leaderboard Score: 0.78468**
- Dataset: 891 passengers, 12 features
