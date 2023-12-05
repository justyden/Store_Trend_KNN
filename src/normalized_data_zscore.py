import pandas as pd
from scipy.stats import zscore

# Step 1: Read the Data
file_path = 'Data/Sample_-_Superstore.xls'
data = pd.read_excel(file_path)

# Step 2: Inspect the Data
data_info = data.info()
data_head = data.head()
missing_values = data.isnull().sum()

# Step 3: Handle Missing Values
# Assuming we want to fill missing numerical values with the mean and categorical with the mode
for column in data.columns:
    if data[column].dtype == 'object':  # Categorical
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:  # Numerical
        data[column].fillna(data[column].mean(), inplace=True)

# Step 4: Normalize Numerical Data using Z-score normalization
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].apply(zscore)

# Save the processed data to a new .xlsx file
processed_file_path = 'Data/Processed_Sample_Superstore.xlsx'
data.to_excel(processed_file_path, index=False)

processed_file_path, data_info, data_head, missing_values

