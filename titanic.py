import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load data 
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 2: Explore 
print("Shape:", df.shape)           # how many rows and columns
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())            # which columns have gaps

# Step 3: Clean
# Fill missing ages with the median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing embarked with most common port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop cabin — too many missing values to be useful
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Convert Sex to numbers (male=0, female=1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert Embarked to numbers
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Step 4: Build features
X = df.drop(columns=["Survived"])  # everything except the answer
y = df["Survived"]                 # the answer column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Train and evaluate 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.1%}")
print(f"F1 Score: {f1_score(y_test, predictions):.1%}")

# Step 6: Interpret 
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature importance:")
print(feature_importance)
