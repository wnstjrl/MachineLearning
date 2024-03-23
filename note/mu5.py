import numpy as np
import matplotlib.pyplot as plt
m=0
std=1
max=10
min=3
step=0.01
x=np.arange(min,max,step)

g1=1/(np.sqrt(np.pi*2)*std)
g2=np.exp(-((x-m)**2)/(2*std**2))
gauss=g1*g2

avg=np.sum(gauss*step)
#step = dx, gauss=f(x)
print("평균:",avg)

plt.plot(x,gauss)
plt.show()
data_size=10000000 #p안정적으로 나오게할라면 늘리기
rv=np.random.randn(data_size)
#randn(): 평균0, 표준편차1 짜리 AWGN발생
p=np.sum(rv>3)/data_size
print(p)