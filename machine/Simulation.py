import numpy as np
import mu12 as cc

data_size = 1024
max_snr = 10
ber = []

for snr_db in range(max_snr - 1, max_snr):
    data = np.random.randint(0, 2, data_size)
    encoded_bit = cc.Encoder(data)

    real_signal = encoded_bit[0, :] * 2 - 1
    imag_signal = encoded_bit[1, :] * 2 - 1

    qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
    ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(data_size)  # 평균파워 1되도록

    noise_std = 10 ** (-snr_db / 20)
    noise = np.random.randn(data_size + 3) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size + 3)
    rcv_signal = np.fft.fft(ofdm_sym) / np.sqrt(data_size) + noise[:data_size + 3]  # noise 크기를 (1027,)로 조정

    real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, data_size + 3)
    imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, data_size + 3)

    # (2, 1024+3)
    dec_input = np.vstack([real_detected_signal, imag_detected_signal])

    decoded_bit = cc.ViterbiDecoder(dec_input)
    print(np.sum(np.abs(dec_input - encoded_bit)))
    print(np.sum(np.abs(data - decoded_bit)))
