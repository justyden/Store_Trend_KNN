''' This updates the excel file to the new format so it can be processed correctly.'''

import pandas as pd

file_path = 'Data\Sample_-_Superstore.xls'
# Read the existing .xls file into a DataFrame
existing_df = pd.read_excel(file_path, engine='xlrd')
existing_df.to_excel('Data\\Updated_Data.xlsx', index=False)