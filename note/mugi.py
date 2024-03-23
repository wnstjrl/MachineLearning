'''과제활용 코드적용
인토더 유징 시프트레지스터 연결구조 바꿀거 인터넷에 코드잇음
시프트레지스터 4개짜리 4개도 표준코드는 잇음 구조바꿔서낼거'''

import numpy as np
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
# [0,0,0,1] [0,0,0,1] 비교해서 1 출력, [0,0,0,1] [0,0,1,1] 비교해서 0.75 출력
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))
# 픽셀 값이 0.5를 넘으면 1로 판정, 이하면 0으로 판정하고, 그 다음 0과 1의 차이를
# 계산하여 비교한 결괄르 출력해줌
stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(100, activation="selu"),
keras.layers.Dense(10, activation="selu"),
])
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[10]),
keras.layers.Dense(28 * 28, activation="sigmoid"),
keras.layers.Reshape([28, 28])
])
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",
optimizer=keras.optimizers.SGD(learning_rate=1.5), metrics=[rounded_accuracy])
history = stacked_ae.fit(X_train, X_train, epochs=20,
validation_data=(X_valid, X_valid))
def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
def show_reconstructions(model, images=X_valid, n_images=5): #이미지 5개만 비교
    reconstructions = model.predict(images[:n_images]) #5개 이미지를 불러와서
    fig = plt.figure(figsize=(n_images * 1.5, 3)) #그림 크기 설정하고
    for image_index in range(n_images):
            plt.subplot(2, n_images, 1 + image_index) #원본 그림
    plot_image(images[image_index])
    plt.subplot(2, n_images, 1 + n_images + image_index) #AutoEncoder 출력
    plot_image(reconstructions[image_index])

#show_reconstructions(stacked_ae)
#plt.show()

from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)
X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())

#plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
#plt.axis("off")
#plt.show()

import matplotlib as mpl
plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
        mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
        position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")
#save_fig("fashion_mnist_visualization_plot")
plt.show()


