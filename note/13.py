import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout


df = fdr.DataReader('005930', '2018-05-04', '2023-05-24') #삼전
print(df.shape)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]
dfx.describe() #모든 값이 0과 1 사이인 것 확인

X = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10
data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size]
    _y = y[i + window_size]
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)

print('전체 데이터의 크기 :', len(data_X), len(data_y))
train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])
test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])
print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=70, batch_size=10)
pred_y =model.predict(test_X)

pred_y = model.predict(test_X)
plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
print("내일 SEC 주가 :", df.Close[-1] * pred_y[-1] / dfy.Close[-1], 'KRW')


# 내일 output 오를지 내릴지 오르면 1 떨어지면0로 표시
# 주가 변동 예측
prediction = model.predict(test_X)
change_direction = np.where(prediction > test_y, 1, 0)

# 그래프 그리기
plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(change_direction, color='blue', label='predicted price direction (1: Up, 0: Down)')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

