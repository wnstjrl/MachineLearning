'''
import numpy as np
import matplotlib.pyplot as plt

std=np.sqrt(0.5) #노이즈 늘리려면 늘리고 줄이려면 줄이기
N=100000 #평균내야해서 많게?아님10개

sig=np.random.randint(2,size=N)*2-1 #numpy randint검색, if문안쓰고 2곱하고 1빼기
noise=std*np.random.randn(N) #가우시안만드는게 렌드앤
#print(np.mean(noise**2))
rcv_sig=sig+noise
dec_sig=(rcv_sig>0)*2-1
#print(sig-dec_sig) 2나-2는 오류난거 N=10일때
error=np.sum(np.abs(sig-dec_sig/2)/N)
print(error)
#노이즈값조절하려면 시그마 조절
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt

N = 100000 # 샘플링 개수
sig = np.random.randint(2, size=N) * 2 - 1 # -1 또는 1 값을 가지는 이진 신호 생성
error_rates = [] # 각 SNR 값에 대한 오류율을 저장할 리스트 생성

for SNR in range(0, 11): # SNR 0~10dB 범위에서 1dB씩 증가
    std = np.sqrt(0.5 / (10 ** (SNR / 10))) # SNR 값에 따라 노이즈의 크기 조절
    noise = std * np.random.randn(N) # 가우시안 노이즈 생성
    rcv_sig = sig + noise # 수신 신호 생성
    dec_sig = (rcv_sig > 0) * 2 - 1 # 수신 신호를 이진 신호로 복조
    error = np.sum(np.abs(sig - dec_sig) / 2) / N # 오류율 계산
    error_rates.append(error) # 오류율을 리스트에 추가
    print("SNR: {}dB, Error rate: {:.6f}".format(SNR, error)) # 결과 출력

plt.plot(range(0, 11), error_rates, 'o-') # 그래프 그리기
plt.xlabel('SNR (dB)')
plt.ylabel('Error rate')
plt.title('Error rate vs. SNR')
plt.grid()
plt.show()
