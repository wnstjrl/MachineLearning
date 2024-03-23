'''
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 덧셈 학습 데이터 생성
np.random.seed(0)
num1 = np.random.randint(0, 100, size=(100000, 1))
num2 = np.random.randint(0, 100, size=(100000, 1))
ans = num1 + num2

# 입력 레이어 정의
inputs = Input(shape=(2,))

# 출력 레이어 정의
outputs = Dense(1)(inputs)

# 모델 생성
model = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 학습
model.fit(x=np.hstack((num1, num2)), y=ans, epochs=10, batch_size=32)
# 덧셈 예측
print(model.predict(np.array([[10, 20]]))) # 출력: [[30.020773]]
'''
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 덧셈 학습 데이터 생성
np.random.seed(0)
num1 = np.random.randint(0, 100, size=(100000, 1))
num2 = np.random.randint(0, 100, size=(100000, 1))
ans = num1 + num2

# 입력 레이어 정의
inputs = Input(shape=(2,))

# 은닉 레이어 1 정의
hidden1 = Dense(64, activation='relu')(inputs)

# 은닉 레이어 2 정의
hidden2 = Dense(32, activation='relu')(hidden1)

# 출력 레이어 정의
outputs = Dense(1)(hidden2)

# 모델 생성
model = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 학습
model.fit(x=np.hstack((num1, num2)), y=ans, epochs=10, batch_size=32)
