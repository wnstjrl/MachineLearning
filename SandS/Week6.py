import numpy as np
import matplotlib.pyplot as plt

t=np.arange(0,1,0.001)
sig=np.sin(2*np.pi*50*t)
plt.figure(1)
plt.plot(sig[0:100])
plt.show()

f_sig=np.fft.fftshift(np.fft.fft(sig))
plt.figure(2)
plt.plot(np.abs(f_sig))
plt.show()

high_sig=sig*np.cos(2*np.pi*200*t)
plt.figure(3)
plt.plot(high_sig[0:100])
plt.show()

f_high_sig=np.fft.fftshift(np.fft.fft(high_sig))
plt.figure(4)
plt.plot(np.abs(f_high_sig))
plt.show()

high_high_sig=high_sig*np.cos(2*np.pi*200*t)
plt.figure(5)
plt.plot(high_high_sig[0:100])
plt.show()

f_high_high_sig=np.fft.fftshift(np.fft.fft(high_high_sig))
plt.figure(6)
plt.plot(np.abs(f_high_high_sig))
plt.show()

print(np.shape(f_high_high_sig))

lpf=np.concatenate((np.zeros((1,400)),np.ones((1,200)),
np.zeros((1,400))),axis=1).squeeze() #squeeze 차원줄이는거 문법적중요x
filtered_f_sig=lpf*f_high_high_sig
plt.figure(7)
plt.plot(np.abs(filtered_f_sig))
plt.show()

final_sig=np.fft.ifft(np.fft.fftshift(filtered_f_sig))
plt.figure(8)
plt.plot(final_sig[0:100])
plt.show()