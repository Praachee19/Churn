import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
df = df.dropna()
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Simple feature engineering
df = pd.get_dummies(df[['tenure', 'MonthlyCharges', 'Contract', 'Churn']], drop_first=True)

# Train/Test split
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "churn_model.pkl")
