import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
print(sys.version)
print(sys.executable)

# Read the data

file_path = 'Data\Sample_-_Superstore.xls'
# Read the existing .xls file into a DataFrame
existing_df = pd.read_excel(file_path, engine='xlrd')
existing_df.to_excel('Data\\Updated_Data.xlsx', index=False)
'''
df = pd.read_excel(file_path)
df.to_excel(file_path, index=False, engine='xlwt')
# This selects the columns to be normalized.
columns_to_normalize = ['Sales', 'Quantity', 'Discount', 'Profit']
scaler = MinMaxScaler() # Normalize using minimum maximum scalar.
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df.to_excel('Data\normalized_data.xls', index=False) # Save the data.
'''