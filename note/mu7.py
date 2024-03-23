import numpy as np
import matplotlib.pyplot as plt

dt=0.001
t=np.arange(0,1,dt)#1000분의1의 간격으로 인덱스만듬
A=1#신호의 크기
f=100#2,000,000,000Hz~3.5GHz
Tb=1#하나의 정보가 차지하는 시간
tx_sig=A*np.cos(2*np.pi*f*t)#송신신호
mat_filter_sig=np.cos(2*np.pi*f*t)#송신신호와 똑같은 필터 만들어야 snr커짐 <-매치드 필터는 수신됐을때 snr최대화
#f100넣어야 매치드필터0.5최대 송신수신신호같아야됨
rcv_sig=np.sum(tx_sig*mat_filter_sig*dt)#dt밑변곱하는거 적분이니 sum싹다더하기 #Integral 적분
# ==> 수신에너지
print(rcv_sig) #수신에너지
#여기까지 중간고사 10시 시험