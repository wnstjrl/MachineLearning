'''
import matplotlib.pyplot as plt
import numpy as np

data_size=1000000
max_snr=11
ber=[] #Bit Error Rate

for snr_db in range(0,max_snr): #range(0~11)==> 0 1 2 ... 10
    signal=np.random.randint(0,2,data_size)*2-1 #1 -1 1 -1 sin -sin sin -sin 디지털일땐 1 0 1 0
    noise_std=10**(-snr_db/20)
    noise=np.random.randn(data_size)*noise_std/np.sqrt(2)
    rcv_signal=signal+noise
    detected_signal=((rcv_signal>0)+0)*2-1
    num_error=np.sum(np.abs(detected_signal-signal))/2
    ber.append(num_error/data_size)
snr=np.arange(0,max_snr)
plt.semilogy(snr,ber)
plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np

data_size=500
max_snr=10 #최대 SNR 13db까지 실험
ber=[]
for snr_db in range(max_snr-1,max_snr):
    real_signal = np.random.randint(0, 2, data_size) * 2 - 1  # -1과 1을 발생
    imag_signal = np.random.randint(0, 2, data_size) * 2 - 1  # -1과 1을 발생

    qpsk_sym=(real_signal+1j*imag_signal)/np.sqrt(2) # 1+j1 1-1j ==> 에너지가 1짜리
    noise_std=10**(-snr_db/20)
    noise=np.random.randn(data_size)*noise_std/np.sqrt(2)+1j*np.random.randn(data_size)*noise_std/np.sqrt(2)
    rcv_signal=qpsk_sym+noise

    real_detected_signal = ((rcv_signal.real > 0) + 0) * 2 - 1
    imag_detected_signal = ((rcv_signal.imag > 0) + 0) * 2 - 1
    num_error=np.sum(np.abs(real_signal-real_detected_signal))/2+np.sum(np.abs(imag_signal-imag_detected_signal))/2
    ber.append(num_error/(data_size*2)) #BER

plt.scatter(rcv_signal.real,rcv_signal.imag)
#snr=np.arange(0,max_snr)
#plt.semilogy(snr,ber)
plt.show()



















