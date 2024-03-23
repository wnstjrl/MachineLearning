'''
import matplotlib.pyplot as plt
import numpy as np

N_sc=64 #subcarrier의 수 LTE:600개~1200개, 5G:1200개
max_snr=20
ber=[]

for snr_db in range(0,max_snr):
    real_signal=np.random.randint(0,2,N_sc)*2-1
    imag_signal=np.random.randint(0,2,N_sc)*2-1
    #데이터가 시간 순서대로 한줄로 오고 있는 상황
    qpsk_sym=(real_signal+1j*imag_signal)/np.sqrt(2)
    ofdm_sym=np.fft.ifft(qpsk_sym)*np.sqrt(N_sc)
    noise_std=10**(-snr_db/20)
    noise = np.random.randn(N_sc)+1j*np.random.randn(N_sc)
    noise = noise*noise_std/np.sqrt(2)
    rcv_sig1=np.fft.fft(ofdm_sym+noise)/np.sqrt(N_sc)
    #rcv_sig2=qpsk_sym+noise
    plt.scatter(rcv_sig1.real,rcv_sig1.imag)
    #plt.scatter(rcv_sig2.real,rcv_sig2.imag)
    plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np

N_sc = 64
max_snr = 14  # SNR 범위를 0~13dB로 변경
num_bits = N_sc * 1000  # 전송할 비트 수

ber = np.zeros(max_snr)  # SNR에 따른 BER 값을 저장할 배열

for snr_idx in range(max_snr):
    snr_db = snr_idx * 1.0
    noise_std = np.ones(N_sc) * 10 ** (-snr_db / 20)
    num_err_bits = 0  # 오류 비트 수
    for i in range(1000):  # 1000번의 전송 시뮬레이션
        # 데이터 신호 생성
        tx_data = np.random.randint(0, 2, size=num_bits)
        # QPSK 변조
        qpsk_sym = (tx_data[::2] * 2 - 1) + 1j * (tx_data[1::2] * 2 - 1)
        # IFFT
        ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(N_sc)
        # 실수 부분을 실제 전송 신호로 간주하고 노이즈 추가
        rcv_sig1 = ofdm_sym + noise_std * np.random.randn(N_sc)
        # FFT
        fft_sym = np.fft.fft(rcv_sig1) / np.sqrt(N_sc)
        # 복호화
        rx_data = np.concatenate([np.real(fft_sym) > 0, np.imag(fft_sym) > 0])
        # 오류 비트 수 계산
        num_err_bits += np.sum(tx_data != rx_data)
    # BER 계산
    ber[snr_idx] = num_err_bits / (num_bits * 1000)

# 그래프 출력
plt.plot(np.arange(max_snr), ber, 'o-')
plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.grid(True)