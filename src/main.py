import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
data = pd.read_excel('../Data/Z_score.xlsx')

# preprocessing
# drop columns that won't be used in the prediction
columns_to_drop = ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                   'Customer ID', 'Country',
                   'Product ID', 'Sales', 'Quantity', 'Discount',
                   'Segment', 'Category', 'Product Name', 'State', 'Region', 'Postal Code', 'Customer Name', ]
data = data.drop(columns=columns_to_drop)

# select the features and target variable
X = data.drop('Sub-Category', axis=1)
# target var
y = data['Sub-Category']

# get unique subcategories and its number from the column
dist_uniq_subcats = data['Sub-Category'].unique()
print("distinct sub-category items:", dist_uniq_subcats, str(data['Sub-Category'].unique()), "counts:", data['Sub-Category'].value_counts())

# encoding categorical variables
categorical_features = ['City']
one_hot_encoder = OneHotEncoder()

# apply the ColumnTransformer to the categorical features using OHE encoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features)],
    remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# initialize the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=700)


# Set hyperparameters for KNeighborsClassifier
n_neighbors_value = 300  # You can set this to your desired value
weights_option = 'uniform'  # or 'distance'  distance
algorithm_option = 'auto'  # or 'ball_tree', 'kd_tree', 'brute'
leaf_size_value = 30 #30
p_value = 2  # This is the power parameter for Minkowski distance
metric_option = 'euclidean'  # Set to 'euclidean' for Euclidean distance

# Create an instance of KNeighborsClassifier with specified hyperparameters
knn = KNeighborsClassifier(n_neighbors=n_neighbors_value,
                           weights=weights_option,
                           algorithm=algorithm_option,
                           leaf_size=leaf_size_value,
                           p=p_value,
                           metric=metric_option)


# fit the classifier to the training data
knn.fit(X_train, y_train)

# predict on the test data
y_pred = knn.predict(X_test)
print("y_test:\n", y_test)
print("y_pred:\n", y_pred)

# evaluate the model
print(classification_report(y_test, y_pred, zero_division=0))

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", conf_matrix)

# create a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=dist_uniq_subcats, columns=dist_uniq_subcats)

# plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted SubCats")
plt.ylabel("true SubCats")
plt.title("confusion matrix")
plt.show()

# print training score
print("training score:\n", knn.score(X_train, y_train) * 100)


# convert the predictions to a DataFrame
predictions_df = pd.DataFrame(y_pred, columns=['predicted Sub-Category'])

# convert the csr_matrix X_test to a dense DataFrame
X_test_df = pd.DataFrame(X_test.toarray())

# concatenate the predictions DataFrame with the dense X_test DataFrame
result_df = pd.concat([X_test_df, predictions_df], axis=1)

# if you also have the actual sub-categories (y_test) to compare against, include them
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
y_test_df.columns = ['actual Sub-Category']

# concatenate actual sub-categories with the result_df
result_df = pd.concat([result_df, y_test_df], axis=1)

print("test vs prediction items:\n", result_df)