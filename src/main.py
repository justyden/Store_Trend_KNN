import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
# data = pd.read_excel('../Data/Z_score.xlsx')
data = pd.read_excel('../Data/Normalized_Data.xlsx')

# preprocessing
# drop columns that won't be used in the prediction
columns_to_drop = ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                   'Customer ID', 'Country',
                   'Postal Code'] #'Sales', 'Quantity', 'Discount', 'Product ID', 'Customer Name', 'Segment', 'Category', 'Product Name', 'State', 'Region',
data = data.drop(columns=columns_to_drop)

# select the features and target variable
X = data.drop('Sub-Category', axis=1)
# target var
y = data['Sub-Category']

# get unique subcategories and its number from the column
dist_uniq_subcats = data['Sub-Category'].unique()
print("distinct sub-category items:", dist_uniq_subcats, str(data['Sub-Category'].unique()), "counts:", data['Sub-Category'].value_counts())

# encoding categorical variables
categorical_features = ['City', 'Segment', 'Category', 'Product Name',
                        'State', 'Region', 'Customer Name', 'Product ID',
                        'Sales', 'Quantity', 'Discount',]
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
classification_report_knn = classification_report(y_test, y_pred, zero_division=0)
print(classification_report_knn)

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


# present data graphically

# bar chart sub category
plt.figure(figsize=(10, 6))
data['Sub-Category'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Sub-Category')
plt.ylabel('Count')
plt.title('Bar Chart of Sub-Category Counts')
plt.show()


# parse classification report string to extract metrics
classification_metrics = classification_report_knn.split('\n')[2:-5]
precision, recall, f1_score, support = [], [], [], []

for metrics in classification_metrics:
    metrics_list = [float(metric) if '.' in metric else int(metric) for metric in metrics.split()[1:]]
    precision.append(metrics_list[0])
    recall.append(metrics_list[1])
    f1_score.append(metrics_list[2])
    support.append(metrics_list[3])

# bar chart for precision
plt.figure(figsize=(12, 6))
plt.bar(dist_uniq_subcats, precision, color='lightblue')
plt.xlabel('Sub-Category')
plt.ylabel('Precision')
plt.title('Precision for Each Sub-Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# bar chart for recall
plt.figure(figsize=(12, 6))
plt.bar(dist_uniq_subcats, recall, color='lightgreen')
plt.xlabel('Sub-Category')
plt.ylabel('Recall')
plt.title('Recall for Each Sub-Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# bar chart for f1 score
plt.figure(figsize=(12, 6))
plt.bar(dist_uniq_subcats, f1_score, color='lightcoral')
plt.xlabel('Sub-Category')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each Sub-Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# bar chart for support
plt.figure(figsize=(12, 6))
plt.bar(dist_uniq_subcats, support, color='grey')
plt.xlabel('Sub-Category')
plt.ylabel('Support')
plt.title('Support for Each Sub-Category')
plt.xticks(rotation=45, ha='right')
plt.show()