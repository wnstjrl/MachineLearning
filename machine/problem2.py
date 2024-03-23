import numpy as np
import matplotlib.pyplot as plt

# QAM symbol mapping
symbol_mapping = {
    (0, 0, 0): -1 - 1j,
    (0, 0, 1): -1 + 1j,
    (0, 1, 0): 1 - 1j,
    (0, 1, 1): 1 + 1j,
    (1, 0, 0): -1j,
    (1, 0, 1): 1j,
    (1, 1, 0): -1,
    (1, 1, 1): 1
}

# Parameters
bits_per_symbol = 3
memory = bits_per_symbol - 1
num_symbols = 1000
snr_range = np.arange(0, 13, 2)
ber_values_viterbi = []
ber_values_viterbi_interleaved = []

for snr in snr_range:
    # Generating Gaussian noise for channel modeling
    noise_power = 1 / (10 ** (snr / 10))
    noise_std_dev = np.sqrt(noise_power / 2)
    noise = noise_std_dev * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))

    # Generating original symbols
    original_bits = np.random.randint(2, size=num_symbols * bits_per_symbol)
    original_bits = original_bits.reshape((num_symbols, bits_per_symbol))
    original_symbols = np.zeros(num_symbols, dtype=complex)
    for i in range(num_symbols):
        original_symbols[i] = symbol_mapping[tuple(original_bits[i])]

    # Channel modeling
    received_symbols = original_symbols + noise

    # Viterbi Decoder
    num_errors_viterbi = 0
    num_errors_viterbi_interleaved = 0

    for i in range(num_symbols):
        # Converting transmitted symbols to QAM symbols
        transmitted_symbol = symbol_mapping[tuple(original_bits[i])]

        # Viterbi decoding
        distances = np.zeros(2 ** memory)
        for state in range(2 ** memory):
            decoder_input = np.array(
                [state >> (memory - j) & 1 for j in range(memory + 1)], dtype=np.uint8
            )

            output_symbol = symbol_mapping[tuple(decoder_input)]
            distances[state] += np.abs(transmitted_symbol - output_symbol) ** 2  # Euclidean distance

        state = np.argmin(distances)
        decoder_input = np.array([state >> (memory - j) & 1 for j in range(memory + 1)], dtype=np.uint8)

        output_bits = np.unpackbits(decoder_input)[-bits_per_symbol:]

        if not np.array_equal(output_bits, original_bits[i]):
            num_errors_viterbi += 1

        # Viterbi decoding with interleaver
        interleaved_bits = np.zeros_like(original_bits[i])
        interleaved_bits[0] = original_bits[i][2]
        interleaved_bits[1] = original_bits[i][0]
        interleaved_bits[2] = original_bits[i][1]

        distances = np.zeros(2 ** memory)
        for state in range(2 ** memory):
            decoder_input = np.array(
                [state >> (memory - j) & 1 for j in range(memory + 1)], dtype=np.uint8
            )

            output_symbol = symbol_mapping[tuple(decoder_input)]
            distances[state] += np.abs(transmitted_symbol - output_symbol) ** 2  # Euclidean distance

        state = np.argmin(distances)
        decoder_input = np.array([state >> (memory - j) & 1 for j in range(memory + 1)], dtype=np.uint8)

        output_bits = np.unpackbits(decoder_input)[-bits_per_symbol:]

        if not np.array_equal(output_bits, interleaved_bits):
            num_errors_viterbi_interleaved += 1

    ber_viterbi = num_errors_viterbi / (num_symbols * bits_per_symbol)
    ber_viterbi_interleaved = num_errors_viterbi_interleaved / (num_symbols * bits_per_symbol)
    ber_values_viterbi.append(ber_viterbi)
    ber_values_viterbi_interleaved.append(ber_viterbi_interleaved)

# Plotting
plt.plot(snr_range, ber_values_viterbi, marker='o', label='Viterbi')
plt.plot(snr_range, ber_values_viterbi_interleaved, marker='o', label='Viterbi with Interleaver')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER Comparison: Viterbi vs Viterbi with Interleaver')
plt.legend()
plt.grid(True)
plt.show()
