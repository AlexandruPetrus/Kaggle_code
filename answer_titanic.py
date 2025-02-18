# Import necessary libraries
import pandas as pd               # For data manipulation
import numpy as np                # For numerical operations
import seaborn as sns             # For data visualization
import matplotlib.pyplot as plt   # For plotting graphs
import missingno as msno          # For visualizing missing data (optional)

# Import machine learning libraries
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LogisticRegression     # For logistic regression modeling
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation

# ---------------------------
# 1. Load the Data
# ---------------------------
# Load the training and testing datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display available columns in the training dataset
print("Available columns in train_df:", train_df.columns.tolist())

# ---------------------------
# 2. Data Cleaning & Preprocessing
# ---------------------------
# Drop columns that are not useful or could cause issues during modeling.
# Check if each column exists before dropping to avoid errors.
cols_to_drop = ['Cabin', 'Ticket', 'Name']
train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], inplace=True)

# Fill missing values in 'Age' with the median value
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the mode (most frequent value)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into numerical ones using one-hot encoding.
# The parameter 'drop_first=True' avoids multicollinearity.
train_df = pd.get_dummies(train_df, drop_first=True)

# Check the columns after transformation
print("Columns after transformation:", train_df.columns.tolist())

# ---------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------
# Display the first few rows of the training dataset
print("Training dataset head:")
print(train_df.head())

# Display the first few rows of the test dataset
print("\nTest dataset head:")
print(test_df.head())

# Display dataset information and descriptive statistics
print("\nTraining dataset info:")
train_df.info()

print("\nTraining dataset descriptive statistics:")
print(train_df.describe())

# Plot the count of survivors
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=train_df)
plt.title("Survivor Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Check for missing values in the training dataset
print("Missing values in the training dataset:")
print(train_df.isnull().sum())

# Visualize missing values using a missingno matrix (optional)
msno.matrix(train_df)
plt.show()

# Plot the distribution of the 'Age' variable
plt.figure(figsize=(8, 4))
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# Plot a heatmap of correlations between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------
# 4. Modeling
# ---------------------------
# Separate the features (X) from the target variable (y)
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))