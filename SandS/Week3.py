import numpy as np #각종배열(벡터)수학계산
import matplotlib.pyplot as plt #그림그리는 라이브러리

'''
time_step=0.1 #간격 과제0.01
t=np.arange(0,10,1) #시간은 0초부터 2초까지 유효한 데이터나오게
x_t=np.exp(-2*t)+1
h_t=np.concatenate(np.ones((1,100)))*time_step #h_t=np.concatenate(np.ones((1,20)))
y_t=np.convolve(x_t,h_t)
plt.figure(1)
plt.stem(x_t)
plt.show()
plt.figure(2)
plt.stem(h_t)
plt.show()
plt.figure(3)
plt.stem(y_t)
plt.show()
'''

import pandas as pd

df=pd.DataFrame([
[1,2,3],
[4,5,6]
], columns=['a', 'b', 'c'], index = ['x', 'y'])

df['d']=df['a']-df['b']

df=df.append(df.sum(), ignore_index=True)
df.index=['x','y','Total']
print(df)