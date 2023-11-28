import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report



# load data from file
data = pd.read_excel('../Data/Normalized_Data.xlsx')
#data = pd.read_csv('../Data/filtered_data.csv')

# set columnes to exclude
drop_columns = ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                'Customer ID', 'Customer Name', 'Country', 'Postal Code',
                'Product ID', 'Product Name', 'Discount', 'Profit']
data = data.drop(columns=drop_columns)
#print(data)

# get unique category values [Sub-Category, Sales, Quantity, Discount, Customer Segment]
unique_segment = data['Segment'].unique()
unique_city = data['City'].unique()
unique_state = data['State'].unique()
unique_region = data['Region'].unique()
unique_catego = data['Category'].unique()
unique_subcat = data['Sub-Category'].unique()
unique_sales = data['Sales'].unique()
unique_quant = data['Quantity'].unique()
#unique_country = data['Country'].unique() # single value not necessary to include

print(f"unique_segment: {unique_segment} \n"
      f"unique_city: {unique_city} \n"
      f"unique_state: {unique_state} \n"
      f"unique_region: {unique_region} \n"
      f"unique_catego: {unique_catego} \n"
      f"unique_subcat: {unique_subcat} \n"
      f"unique_sales: {unique_sales} \n"
      f"unique_quant: {unique_quant} \n")

# # verify data from OHE
# columns = ohe.get_feature_names_out(one_hot_encoding_fields)
# encoded_df = pd.DataFrame(enc_data.toarray(), columns=columns)
# #print(encoded_df.head())

# separate data to test and train
X = data.drop('Sub-Category', axis=1)
y = data['Sub-Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)


# preprocessing, convert categorical features to numerical for use in distance formula
ohe_fields = ['Segment', 'State', 'Region', 'Category']
ohe = OneHotEncoder(sparse_output=False)

X_train_ohe = ohe.fit_transform(X_train[ohe_fields])
X_test_ohe = ohe.transform(X_test[ohe_fields])


# LE encoding sub-cat, city
scaler = MinMaxScaler()
le_sub = LabelEncoder()
le_city = LabelEncoder()

y_train_encoded = le_sub.fit_transform(y_train)
y_test_encoded = le_sub.transform(y_test)

X_train['City'] = le_city.fit_transform(X_train['City'])
X_test['City'] = le_city.transform(X_test['City'])

X_train[['City']] = scaler.fit_transform(X_train[['City']])
X_test[['City']] = scaler.transform(X_test[['City']])


# print("data['Sub-Category']:", data['Sub-Category'])
# print("data['City']:", data['City'])


# convert OHE encoded fields to dataframe

# Convert OHE encoded fields to DataFrame
X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names_out(ohe_fields), index=X_train.index)
X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names_out(ohe_fields), index=X_test.index)


# combine OHE encoded fields with numerical

X_train_final = pd.concat([X_train_ohe_df, X_train['City'], X_train['Sales'], X_train['Quantity']], axis=1)
X_test_final = pd.concat([X_test_ohe_df, X_test['City'], X_test['Sales'], X_test['Quantity']], axis=1)

#
# comb_data = pd.concat([ohe_fields_df, data['Sub-Category'], data['City']], axis=1)
# print(f"comb_data: {comb_data}")
#
# print(comb_data.columns)


# run kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_final, y_train_encoded)

# predict
y_pred = knn.predict(X_test_final)

print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le_sub.classes_))

import matplotlib.pyplot as plt

# Scatter plot of Actual vs Predicted values
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Sales')
# plt.ylabel('Predicted Sales')
# plt.title('Actual vs Predicted Sales')
# plt.show()