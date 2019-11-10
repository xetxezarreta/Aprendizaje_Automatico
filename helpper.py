'''
General purpose
'''
# read dataframe
import pandas as pd
df = pd.read_csv('path')

# divide df on features/target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# get df description
df.describe()

# check target imbalance
y.value_counts()

# check df missing values
df.isnull().values.any()

'''
Classification
'''
# K-Nearest Neighbors