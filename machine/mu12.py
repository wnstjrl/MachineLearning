import numpy as np

def Encoder(data):
    data=np.append(data,[0,0,0]) #Shift Register 설정 및 초기화
    dataSize=np.shape(data)[0] #(64,)
    shiftReg=[0,0,0] #K=3
    encoded_bit=np.zeros((2,dataSize))

    for i in range(dataSize):
        shiftReg[2]=shiftReg[1]
        shiftReg[1]=shiftReg[0]
        shiftReg[0]=data[i]
        encoded_bit[0,i]=np.logical_xor(np.logical_xor(shiftReg[0],shiftReg[1]),shiftReg[2])
        encoded_bit[1,i]=np.logical_xor(shiftReg[0],shiftReg[2])

    return encoded_bit

def ViterbiDecoder(encoded_bit):
    ref_out=np.zeros((2,8))
    ref_out[0,:]=[0,1,1,0,1,0,0,1]
    ref_out[1,:]=[0,1,0,1,1,0,1,0]
    #00/01/10/11 들어오는 화살표의 출력들
    #00으로 들어으는 것의 과거 state 00과 01
    #11으로 들어오는 것의 과거 state 10과 11
    dataSize=np.shape(encoded_bit)[1] #2 by 원래데이터길이+3[0,0,0]
    cumDist=[0,100,100,100]#초기값 설정 00/01/10/11
    prevState=[]
    for i in range(dataSize):
        tmpData=np.tile(encoded_bit[:,i].reshape(2,1),(1,8))
        # 0 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0 0
        dist=np.sum(np.abs(tmpData-ref_out),axis=0)

        tmpDist=np.tile(cumDist,(1,2))+dist

        tmpPrevState=[]
        for a in range(4): #state 수가 4이니까
            if tmpDist[0,2*a+0]<=tmpDist[0,2*a+1]:
                cumDist[a]=tmpDist[0,2*a+0]
                tmpPrevState.append((a%2)*2+0)
            else:
                cumDist[a]=tmpDist[0,2*a+1]
                tmpPrevState.append((a%2)*2+1)
        prevState.append(tmpPrevState)

        state_index=np.argmin(cumDist)
        #print(state_index, cumDist)
        # 00 01 10 11 / cD: 0 2 3 2
        decoded_bit=[]
    for b in range(dataSize -1,-1,-1):#디코딩 과정은 역순
        decoded_bit.append(int(state_index/2))
        state_index=prevState[b][state_index]
    data_size=np.shape(decoded_bit)[0]
    decoded_bit=np.flip(decoded_bit)[0:data_size-3]
    return decoded_bit