import numpy as np
import matplotlib.pyplot as plt

t=np.arange(0,1,0.0001) #샘플링
f1=1000
f2=2500
f3=4000

sig1=3*np.cos(2*np.pi*f1*t)
sig2=2*np.cos(2*np.pi*f2*t)
sig3=1*np.cos(2*np.pi*f3*t)
sig=sig1+sig2+sig3
fft_sig=np.fft.fftshift(np.fft.fft(sig))/10000
#plt.plot(np.abs(fft_sig))
#plt.show()

# 1000Hz: 5000Hz 최대표현주파수==>pi/5
# 2500Hz: 5000Hz 최대표현주파수==>pi/2
# 4000Hz: 5000Hz 최대표현주파수==>4*pi/5

f_phase1=np.pi/6 #1666.66Hz 0~pi ==> 0~5000Hz
p=0.95 #Pole반지름
z=1 #Zero반지름

pole=p*np.exp(1j*f_phase1)
zero=z*np.exp(1j*(np.pi-f_phase1))

f=np.arange(0,6.28,0.000628)-3.14 #z=np.exp(1j*w), w=0~pi
D_z=1-2*pole.real*np.exp(-1*1j*f)+p*p*np.exp(-2*1j*f)
N_z=1-2*zero.real*np.exp(-1*1j*f)+z*z*np.exp(-2*1j*f)
T_f=N_z/D_z
freq=np.arange(-5000,5000,1)
plt.plot(freq,np.abs(T_f),freq,np.abs(fft_sig))
#plt.hold(True)
#plt.plot(np.abs(fft_sig))
plt.show()
