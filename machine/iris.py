import numpy as np
import sklearn
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns

iris = sns.load_dataset("iris")
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
plt.show()
sns.scatterplot(x="petal_length", y="petal_width", hue="species", data=iris)
plt.show()

iris_data = load_iris() #아이리스 데이터 불러오기
#print(iris_data.data) #아이리스 데이터 나타내기
data = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
data['class'] = iris_data.target #data,class,feature_names 이용하여 DataFrame생성
print(data)
print(data.describe())

X = data[data.columns[0:4]]
y = data[['class']]
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, random_state=42,test_size = 0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

ss =StandardScaler()
ss.fit(X_train)
ss_train = ss.transform(X_train)
ss_test = ss.transform(X_test)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

from sklearn.linear_model import LogisticRegression #로지스틱회귀분석
model=LogisticRegression()
logit = LogisticRegression()
logit.fit(ss_train, y_train)
pred_y_train = logit.predict(ss_train)
pred_y_test = logit.predict(ss_test)
model.fit(ss_train, y_train)
print("model score train : ",model.score(ss_train, y_train))
print("model score test : ",model.score(ss_test, y_test))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred_y_test))

from sklearn.tree import DecisionTreeClassifier #의사결정나무
tree = DecisionTreeClassifier()
tree.fit(ss_train, y_train)
pred_y_train = tree.predict(ss_train)
pred_y_test = tree.predict(ss_test)
print("model score train : ",tree.score(ss_train, y_train))
print("model score test : ",tree.score(ss_test, y_test))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred_y_test))

from sklearn.ensemble import RandomForestClassifier #랜덤포레스트
random = RandomForestClassifier()
random.fit(ss_train, y_train)
pred_y_train = random.predict(ss_train)
pred_y_test = random.predict(ss_test)
print("model score train : ",random.score(ss_train, y_train))
print("model score test : ",random.score(ss_test, y_test))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred_y_test))

from sklearn.neighbors import KNeighborsClassifier #최근접이웃
knn = KNeighborsClassifier()
knn.fit(ss_train,y_train)
pred_y_train = knn.predict(ss_train)
pred_y_test = knn.predict(ss_test)
print("model score train : ",knn.score(ss_train, y_train))
print("model score test : ",knn.score(ss_test, y_test))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred_y_test))