from matplotlib import pyplot as plt
import numpy as np
#신호 ==> cos 곱해주고
t=np.arange(0,1,0.001) #0.0001초 간격으로 10000개
sig=1*np.sin(50*t)-2*np.sin(100*t-0.01)+3*np.sin(150*t+0.05)
plt.subplot(711)
plt.plot(t,sig)

f_sig=np.fft.fft(sig)
f_sig=np.fft.fftshift(f_sig)
plt.subplot(712)
plt.plot(np.abs(f_sig))

carrier=np.cos(2*np.pi*200*t) # 2*pi*f*t
plt.subplot(713)
plt.plot(t,carrier)

am_sig=sig*carrier
plt.subplot(714)
plt.plot(t,am_sig)

f_am_sig=np.fft.fft(am_sig)
f_am_sig=np.fft.fftshift(f_am_sig)
plt.subplot(715)
plt.plot(np.abs(f_am_sig))

am_sig=am_sig*carrier
plt.subplot(716)
plt.plot(t,am_sig)

f_am_sig=np.fft.fft(am_sig)
f_am_sig=np.fft.fftshift(f_am_sig)
plt.subplot(717)
plt.plot(np.abs(f_am_sig))


plt.show()