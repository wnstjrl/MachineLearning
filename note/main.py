import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

bream_data_size=60
smelt_data_size=20
#도미 데이터
bream_length = np.random.randn(1,bream_data_size)*5+35
bream_weight = bream_length*20+np.random.randn(1,bream_data_size)*20
#빙어 데이터
smelt_length= np.random.randn(1,smelt_data_size)*2+12
smelt_weight= smelt_length+np.random.randn(1,smelt_data_size)*2


length=np.concatenate((bream_length,smelt_length),axis=1)
weight=np.concatenate((bream_weight,smelt_weight),axis=1)

fish_data = np.concatenate((length,weight),axis=0).transpose()#입력
fish_target = np.array([1]*bream_data_size + [0]*smelt_data_size)#출력

mean=np.mean(fish_data,axis=0) #(80,2) ==> 세로로 평균내서 2개의 평균값
std=np.std(fish_data,axis=0)
fish_data=(fish_data-mean)/std #Normalization 정규화

np.random.seed(42)
index=np.arange(bream_data_size+smelt_data_size)
np.random.shuffle(index)

train_input=fish_data[index[0:60]]
train_output=fish_target[index[0:60]]

test_input=fish_data[index[60:]]
test_output=fish_target[index[60:]]

kn = KNeighborsClassifier() #학습 알고리즘
kn.fit(train_input, train_output) #컴퓨터가 학습 Mapping
print(kn.score(test_input, test_output))
#[25,250] => List
test_in=(np.array([25, 250]).reshape(1,2)-mean)/std
print('결과는:', kn.predict(test_in))

plt.figure(3)
plt.scatter(fish_data[:,0], fish_data[:,1])
print(np.shape(fish_data[:,0]))
plt.scatter(test_in[:,0], test_in[:,1], marker='^')

plt.xlabel('length')
plt.ylabel('weight')
plt.show()