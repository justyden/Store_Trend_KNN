''' This script normalizes the data.'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
print(sys.version)
print(sys.executable)

# Read the data
file_path = 'Data\\Updated_Data.xlsx'
df = pd.read_excel(file_path)
# This selects the columns to be normalized.
columns_to_normalize = ['Sales', 'Quantity', 'Discount', 'Profit']
scaler = MinMaxScaler()
# Normalize the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
# Save the normalized data to a new file
df.to_excel('Data\\Normalized_Data.xlsx', index=False)
