import pandas as pd
import numpy as np
import os
#gets direcftory
os.getcwd()
#loads data
df = pd.read_csv('./data/insurance.csv')

df.head()

##Create dummy variables
regions = pd.get_dummies(df[['region']])
#removes text region
df.drop('region',inplace=True,axis=1)
#joints region dummies with dataframe
df1 = pd.DataFrame.join(df,regions)
#creates dummies from smoker variable
def smoker(self):
    if self =='yes':
        return 1
    else:
        return 0
#creates gender variable 0=female 1=male
def gender(self):
    if self =='male':
        return 1
    else:
        return 0

#applies both functions
df1['smoker'] = df1['smoker'].apply(smoker)

df1['sex'] = df1['sex'].apply(gender)

#writes the transformed data to a new file
df1.to_csv('./data/insurance_transformed.csv')