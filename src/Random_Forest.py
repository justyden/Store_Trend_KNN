import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_excel('../Data/Z_score.xlsx')
threshold_value = 0.776255947
negative_pro = -0.117632811



# Convert 'Profit' to a categorical variable (example: high profit vs. low profit vs negative profit)
# You need to define the logic for this conversion
data['Profit_Category'] = data['Profit'].apply(lambda x: 'High' if x > threshold_value else ('Negative' if x <=negative_pro else 'Low'))


# Selecting only the relevant columns
data = data[['Category', 'Sales', 'Quantity', 'Sub-Category','Discount', 'Profit_Category']]

# Preprocessing - One-Hot Encoding for categorical features
categorical_features = ['Sub-Category', 'Category']
one_hot_encoder = OneHotEncoder(sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features)],
    remainder='passthrough')

y = data['Profit_Category']
X = data.drop('Profit_Category', axis=1)

X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=250, max_depth=20, random_state=100)

# Fit the classifier to the training data
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)
# predict on the test data
print("y_test:\n", y_test)
print()
print("y_pred:\n", y_pred)

# Evaluate the model using classification metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", conf_matrix)


# print training score
print("training score:\n", rf.score(X_train, y_train)* 100)


