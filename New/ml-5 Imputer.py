from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

s1=np.array(['34','65','34',np.nan,'43','54',np.nan,'23','54','65','34',np.nan])
s1.shape=(-1,1)

imp = SimpleImputer(strategy='most_frequent')
imp.fit(s1)		


#Filling Data
s1=imp.transform(s1)

print(s1)
