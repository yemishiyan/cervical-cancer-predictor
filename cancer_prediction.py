import numpy as np  # Add this line
import pandas as pd  # Already included
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
print("Current Working Directory:", os.getcwd())  # Ensure you're in the correct folder

import pandas as pd

file_path = "risk_factors_cervical_cancer.csv"  # Ensure it's the correct filename

df = pd.read_csv(file_path)

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert numerical columns from object type to proper numeric format
for col in df.columns:
    if df[col].dtype == "object":  # Check if column is object type
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, set errors to NaN

# Fill missing values with the column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Verify the data types
print(df.dtypes)
print("Missing values handled successfully!")
print("Dataset Loaded Successfully!\n")
print(df.head())  # Show first few rows
print(df.info())  # Get details about data types and missing values
try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!\n")
    print(df.head())  # Show first few rows
except FileNotFoundError:
    print("Error: Dataset not found! Check if the file name and path are correct.")


import numpy as np

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Check the number of missing values per column
print(df.isnull().sum())

# Fill missing numerical values with the column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding to categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print("Categorical Data Encoded Successfully!")

from sklearn.utils import resample

# Separate majority & minority classes
df_majority = df[df['Dx:Cancer'] == 0]  # Low Risk
df_minority = df[df['Dx:Cancer'] == 1]  # High Risk

# Make sure minority class is actually upsampled!
df_minority_upsampled = resample(df_minority, replace=True, n_samples=840, random_state=42)

# Combine and shuffle the new dataset
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
from sklearn.model_selection import train_test_split

# Define features & target
X = df_balanced.drop(columns=['Dx:Cancer'])  # Features
y = df_balanced['Dx:Cancer']  # Target variable

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm shape of datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler

# Apply scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with scaled data
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# Test model performance again
y_log_pred = log_model.predict(X_test_scaled)
print(f"New Model Accuracy (with scaling): {accuracy_score(y_test, y_log_pred):.2f}")

print("New Class Distribution:")
print(df_balanced['Dx:Cancer'].value_counts())  # Should show 840 cases for both 0 and 1

print("Dataset balanced successfully!")
print(df_balanced['Dx:Cancer'].value_counts())  # Verify new class balance

X = df_balanced.drop(columns=['Dx:Cancer'])  # Features
y = df_balanced['Dx:Cancer']  # Target variable


print(df.dtypes)  # This prints the column types in the terminal
df = df.apply(pd.to_numeric, errors="coerce")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split successfully!")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(class_weight="balanced", max_iter=500)
log_model.fit(X_train, y_train)

y_log_pred = log_model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_log_pred):.2f}")
# Train the model
log_model.fit(X_train, y_train)

# Evaluate new predictions
y_pred = log_model.predict(X_test)

print(f"New Model Accuracy: {accuracy_score(y_test,y_pred):.2f}")


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Calculate feature importance using Logistic Regression coefficients
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': log_model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Now `importance` is defined
importance = feature_importance["Importance"]

print(feature_importance)

# Create a DataFrame for visualization
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_df['Feature'], y=feature_df['Importance'])
plt.xticks(rotation=90)
plt.title("Feature Importance in Predicting Cervical Cancer Risk")
plt.show()

from sklearn.metrics import confusion_matrix, classification_report

# Get confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Get precision, recall, and F1-score
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(log_model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_log_pred = log_model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_log_pred):.2f}")

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_gb_pred = gb_model.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_gb_pred):.2f}")

import joblib

# Save the trained model
joblib.dump(log_model, "cancer_risk_predictor.pkl")
print("Model saved successfully!")


# Shuffle dataset to mix up cases
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset balanced successfully! New class counts:")
print(df_balanced['Dx:Cancer'].value_counts())

print(df_balanced.head())  # See the first few rows
print("Training Data Class Distribution:")
print(y_train.value_counts())  # Check if both classes exist in training set

test_input = pd.DataFrame([[45, 10, 18, 1] + [1] * (len(log_model.feature_names_in_) - 4)], 
                          columns=log_model.feature_names_in_)

print("Test Prediction for High-Risk Case:", log_model.predict(test_input))

test_inputs = pd.DataFrame([
    [45, 10, 18, 1] + [1] * (len(log_model.feature_names_in_) - 4),  # High-risk case
    [30, 2, 16, 0] + [0] * (len(log_model.feature_names_in_) - 4)   # Low-risk case
], columns=log_model.feature_names_in_)

print("Test Predictions:", log_model.predict(test_inputs))

import joblib

# Save trained model
joblib.dump(log_model, "cancer_risk_predictor.pkl")