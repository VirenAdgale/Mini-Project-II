import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")  # replace with your file path

# Drop columns that don't help or leak info
df = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

# Encode 'type' feature (CASH-IN, CASH-OUT, etc.)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Features and target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)