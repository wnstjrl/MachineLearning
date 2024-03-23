import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

ep=np.array(pd.read_excel('ep.xlsx'))
fp=np.array(pd.read_excel('fp.xlsx'))

print(np.shape(ep))
print(np.shape(fp))
plt.plot(fp[:,0])
plt.show()

print(np.shape(ep))
print(np.shape(np))

fp=fp[:,0:3]


from sklearn.preprocessing import PolynomialFeatures


poly=PolynominalFeatures(degree=3,include_bias=False)
poly.fit(fp)
train_poly=poly.transform(fp)
print(np.shape(train_poly))

tr_in, ts_in, tr_out, ts_out=train_test_split(train_poly, ep, test_size=0.50,random_state=42)
lr=LogisticRegression()