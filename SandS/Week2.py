import numpy as np #각종배열(벡터)수학계산
import matplotlib.pyplot as plt #그림그리는 라이브러리

signal=[0,0,0,0,0,0,1,1,1,1,1,0,0,0]
system=[0,0,0,0,0,0,1,1,1,1,1,0,0,0]
output=np.convolve(signal,system)
output2=output
output3=np.convolve(output,output2)
delta=[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
plt.figure(1)
plt.stem(delta)
plt.figure(2)
plt.stem(np.convolve(output3,delta))
plt.show()