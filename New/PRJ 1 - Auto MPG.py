#Augo MPG

import pandas as pd
from sklearn.linear_model import LinearRegression

columns=['mpg','cylinders','displacement','horsepower','weight','accelaration','year','origin','carname']
data=pd.read_csv(r"‪‪C:\Users\girish\Desktop\auto mpg.csv", header=None, skiprows=1, names=columns)

print(data.head())
print(data.tail())


#Check for Nan Values
print(data.isna().sum())
