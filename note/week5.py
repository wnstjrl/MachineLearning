import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#DataFrame ==> np ==> 표준화(평균을 중심으로 동일한 표준편차)
wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(train_scaled,train_target)

print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
print(lr.coef_,lr.intercept_)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled,train_target)

print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=1, filled=True,
          feature_names=['alcohol','sugar','pH'])
plt.show()

dt=DecisionTreeClassifier(max_depth=3,random_state=42)
dt.fit(train_scaled,train_target)

print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))
plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,
          feature_names=['alcohol','sugar','pH'])
plt.show()

dt=DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input,train_target)
print(dt.score(train_input,train_target))
print(dt.score(test_input,test_target))
plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,
          feature_names=['alcohol','sugar','pH'])
plt.show()
print(dt.feature_importances_)

'''
from sklearn.model_selection import cross_validate
scores=cross_validate(dt,train_input,train_target)
print(scores)
import numpy as np
print(np.mean(scores['test_score']))


from sklearn.model_selection import GridSearchCV
params={'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}
gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs.fit(train_input,train_target)
dt=gs.best_estimator_
print(dt.score(train_input,train_target))
print(gs.best_params_)
{'min_impurity_decrease':0.0001}
print(gs.cv_results_['mean_test_score'])

from scipy.stats import uniform, randint
rgen=randint(0,10)
rgen.rvs
'''