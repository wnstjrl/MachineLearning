import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

# 데이터셋 불러오기
df = pd.read_csv('aptos2019-blindness-detection/train.csv')

# 이미지 경로와 등급 정보 추출
image_paths = df['id_code'].values
labels = df['diagnosis'].values

# 이미지 데이터 로드 함수 및 전처리
def load_image(image_path):
    # 이미지 로드
    image = tf.keras.preprocessing.image.load_img('aptos2019-blindness-detection/train_images/' + image_path + '.png',
                                                  target_size=(224, 224))
    # 이미지를 numpy 배열로 변환
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    # 이미지 전처리 (예시: 정규화)
    preprocessed_image = image_array / 255.0  # 0과 1 사이로 정규화
    return preprocessed_image

# 이미지 데이터와 레이블 생성
images = np.array([load_image(path) for path in image_paths])
labels = np.array(labels)

# 데이터 분할
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# CNN 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.BatchNormalization(),  # 정규화를 위한 BatchNormalization 레이어 추가
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

# 모델 학습
history = model.fit(train_images, train_labels, epochs=30, batch_size=32, validation_data=(test_images, test_labels), callbacks=[early_stopping])

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 예측 결과 확인
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted Labels:", predicted_labels)
print("True Labels:", test_labels)
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy:", accuracy)

# Loss 그래프 출력
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
