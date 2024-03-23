import matplotlib.pyplot as plt
import numpy as np

data_size = 128  # 하나의 OFDM 심볼에 전송되는 QPSK 심볼의 수
max_snr = 13  # 최대 SNR 13dB까지 실험
max_repeat = 10000
ber = []

for snr_db in range(0, max_snr):
    tmp_error_rate = 0
    for _ in range(max_repeat):
        data = np.random.randint(0, 2, data_size)  # 0과 1을 발생

        # 채널 인코딩 및 QPSK 변조
        enc_data = channel_encoder(data)
        real_signal = enc_data[0, :] * 2 - 1
        imag_signal = enc_data[1, :] * 2 - 1
        qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)

        # OFDM 변조
        ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(data_size)

        # 노이즈 추가
        noise_std = 10 ** (-snr_db / 20)
        noise = np.random.randn(data_size) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size) * noise_std / np.sqrt(2)
        rcv_signal = ofdm_sym + noise

        # 수신된 OFDM 심볼 처리
        rcv_signal = np.fft.fft(rcv_signal) / np.sqrt(data_size)
        dec_real_signal = (rcv_signal.real > 0)
        dec_imag_signal = (rcv_signal.imag > 0)
        dec_signal = np.vstack((dec_real_signal, dec_imag_signal))

        # 채널 디코딩
        decoded_data = channel_decoder(dec_signal)
        error = np.sum(np.abs(data - decoded_data))
        tmp_error_rate += error

    error_rate = tmp_error_rate / (data_size * max_repeat)
    ber.append(error_rate)

# BER 그래프 출력
plt.plot(range(max_snr), ber)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.show()
