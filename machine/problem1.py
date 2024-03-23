import matplotlib.pyplot as plt
import numpy as np

data_size = 128
max_snr = 13
max_repeat = 10000
ber_no_coding = []
ber_with_coding = []

for snr_db in range(0, max_snr):
    tmp_error_rate_no_coding = 0
    tmp_error_rate_with_coding = 0

    for _ in range(max_repeat):
        data = np.random.randint(0, 2, data_size)

        real_signal_no_coding = (2 * data - 1)  # Convert to -1 and 1
        imag_signal_no_coding = np.zeros(data_size)  # Initialize with zeros

        qpsk_sym_no_coding = (real_signal_no_coding + 1j * imag_signal_no_coding) / np.sqrt(2)
        ofdm_sym_no_coding = np.fft.ifft(qpsk_sym_no_coding) * np.sqrt(data_size)

        noise_std = 10 ** (-snr_db / 20)
        noise = np.random.randn(data_size) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size) * noise_std / np.sqrt(2)
        rcv_signal_no_coding = ofdm_sym_no_coding + noise
        rcv_signal_no_coding = np.fft.fft(rcv_signal_no_coding) / np.sqrt(data_size)

        dec_real_signal_no_coding = (rcv_signal_no_coding.real > 0)
        dec_imag_signal_no_coding = (rcv_signal_no_coding.imag > 0)
        dec_signal_no_coding = np.vstack((dec_real_signal_no_coding, dec_imag_signal_no_coding))

        decoded_data_no_coding = np.zeros(data_size, dtype=int)
        for i in range(data_size):
            decoded_data_no_coding[i] = dec_signal_no_coding[0, i]

        error_no_coding = np.sum(np.abs(data - decoded_data_no_coding))
        tmp_error_rate_no_coding += error_no_coding

    error_rate_no_coding = tmp_error_rate_no_coding / (data_size * max_repeat)
    ber_no_coding.append(error_rate_no_coding)

    # Channel Coding
    tmp_error_rate_with_coding = 0
    for _ in range(max_repeat):
        data = np.random.randint(0, 2, data_size)

        # Coding
        coded_data = np.zeros(data_size * 2, dtype=int)
        for i in range(data_size):
            coded_data[i * 2] = data[i]
            coded_data[i * 2 + 1] = 1 - data[i]  # Complement

        # Noisy Channel
        noise = np.random.randn(data_size * 2) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size * 2) * noise_std / np.sqrt(2)
        rcv_signal_with_coding = coded_data + noise

        # Decoding
        decoded_data_with_coding = np.zeros(data_size, dtype=int)
        for i in range(data_size):
            decoded_data_with_coding[i] = 1 if rcv_signal_with_coding[i * 2] < rcv_signal_with_coding[i * 2 + 1] else 0

        error_with_coding = np.sum(np.abs(data - decoded_data_with_coding))
        tmp_error_rate_with_coding += error_with_coding

    error_rate_with_coding = tmp_error_rate_with_coding / (data_size * max_repeat)
    ber_with_coding.append(error_rate_with_coding)

# BER Comparison Graph
snr_range = range(0, max_snr)
plt.plot(snr_range, ber_no_coding, label='No Coding')
plt.plot(snr_range, ber_with_coding, label='With Coding')

plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER Comparison of OFDM-QPSK with and without Channel Coding')
plt.legend()
plt.grid(True)
plt.show()
