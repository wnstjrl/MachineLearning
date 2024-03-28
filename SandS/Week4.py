import numpy as np
import matplotlib.pyplot as plt
'''
time_step=0.01
t=np.arange(0,1,time_step).reshape(1,100)

fs=np.zeros((1,100))
rec=np.concatenate((np.ones((1,50)),-np.ones((1,50))),axis=1)
count=1

for f in range(1,12,2):
fs+=np.sin(2*np.pi*(f)*t)/count
plt.subplot(6,1,count)
count+=1
plt.plot(fs.T)
plt.plot(rec.T)
plt.show()
'''


import numpy as np
import matplotlib.pyplot as plt
k=np.arange(-20,21,1)
print(k)
T=2
t=1
fs=np.sin(np.pi*t*k/T)/(np.pi*k)
plt.figure(1)
plt.stem(fs)
plt.show()