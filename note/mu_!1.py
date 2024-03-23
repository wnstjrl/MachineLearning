'''
import matplotlib.pyplot as plt
import numpy as np

N_sc=64 #subcarrier 수 LTE:600~1200개 5G 1200개
max_snr=20
ber=[]

for snr_db in range(0,max_snr):
    real_signal = np.random.randint(0, 2, N_sc) * 2 - 1
    imag_signal = np.random.randint(0, 2, N_sc) * 2 - 1
    #데이터가 시간 순서대로 한줄로 오고 있는 상황
    qpsk_sym=(real_signal+1j*imag_signal)/np.sqrt(2) #루트2로 나눠줘야 1나옴
    #print(np.abs(qpsk_sym)**2)/N_sc)
    ofdm_sym=np.fft.ifft(qpsk_sym)
    #print(np.abs(ofdm_sym) ** 2) / N_sc)
    noise_std=10**(-snr_db/20)
    noise=np.random.randn(N_sc)+1j*np.random.randn(N_sc)
    noise=noise*noise_std/np.sqrt(2)
    rcv_sig1 = np.fft.fft(ofdm_sym + noise)
    #이사이 for문 넣야됨
    #print(np.sum(np.abs(rcv_sig1)**2)/N_sc)
    #rcv_sig1 = ofdm_sym + noise
    #rcv_sig2=qpsk_sym+noise
    plt.scatter(rcv_sig1.real,rcv_sig1.imag)
    #plt.scatter(rcv_sig2.real, rcv_sig2.imag)
    plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np

N_sc = 64  # subcarrier 수 LTE:600~1200개 5G 1200개
max_snr = 13
n_trials = 1000  # number of Monte Carlo trials for each SNR level
ber = np.zeros(max_snr+1)

for snr_db in range(max_snr+1):
    n_errors = 0
    n_bits = 0
    snr_linear = 10**(snr_db/10)
    noise_std = 1 / np.sqrt(2 * snr_linear)  # calculate noise standard deviation
    for i in range(n_trials):
        # generate random QPSK symbols and modulate with OFDM
        qpsk_sym = (np.random.randint(0, 2, N_sc) * 2 - 1) / np.sqrt(2)
        ofdm_sym = np.fft.ifft(qpsk_sym)
        # add complex Gaussian noise to the OFDM symbol
        noise = np.random.randn(N_sc) + 1j * np.random.randn(N_sc)
        noise *= noise_std
        rcv_sig = np.fft.fft(ofdm_sym + noise)
        # demodulate the received signal and count errors
        rcv_qpsk = rcv_sig / np.sqrt(N_sc)
        rcv_bits = np.where(rcv_qpsk.real >= 0, 1, 0)
        rcv_bits = np.concatenate((rcv_bits, np.where(rcv_qpsk.imag >= 0, 1, 0)))
        n_errors += np.sum(np.abs(rcv_bits - np.round(qpsk_sym.real)) + np.abs(rcv_bits - np.round(qpsk_sym.imag)))
        n_bits += N_sc * 2
    ber[snr_db] = n_errors / n_bits

plt.plot(np.arange(max_snr+1), ber, 'o-')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.show()



