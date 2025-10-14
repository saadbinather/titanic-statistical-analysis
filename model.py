# ---------------------------------------------
# 🛳️ Titanic Survival Prediction (Complete Project)
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
# ---------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------
# Step 2: Load Dataset
# (Make sure you have 'titanic.csv' in your working directory)
df = pd.read_csv('titanic.csv')

# Take a quick look
print("🔹 First 5 rows:")
print(df.head())

# ---------------------------------------------
# Step 3: Basic Cleaning
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# Fill missing 'Age' with mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing 'Embarked' with most common value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' because it has too many missing values
df.drop(['Cabin'], axis=1, inplace=True)

# ---------------------------------------------
# Step 4: Drop Unnecessary Columns
# These columns are not helpful for prediction
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# ---------------------------------------------
# Step 5: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 4))
sns.countplot(x='Survived', data=df)
plt.title("Overall Survival Count (0 = No, 1 = Yes)")
plt.show()

plt.figure(figsize=(10, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(10, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()



# Step 6: Feature Encoding
# Convert 'Sex' to numbers using LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male→1, female→0

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# Step 7: Split Dataset
X = df.drop('Survived', axis=1)  # features
y = df['Survived']               # target label

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Step 8: Train Models

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


# Step 9: Evaluate Models
print("\n🔹 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

print("\n🔹 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# Step 10: Feature Importance (Random Forest)
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Survival Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# Step 11: Survival Trends
plt.figure(figsize=(8, 4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Fare', hue='Survived', multiple='stack')
plt.title("Fare Distribution by Survival")
plt.show()

print("\n✅ Titanic Survival Prediction Complete!")
