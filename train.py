import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 데이터 로드 및 전처리 함수
def load_keypoint_data(folder_path):
    keypoints = []
    labels = []
    max_length = 0

    # 모든 파일을 읽어서 가장 긴 키포인트 길이 확인
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder_path, file))
            keypoints.append(data)
            labels.append(0)  # 정상 행동 레이블
            if data.shape[0] > max_length:
                max_length = data.shape[0]

    # 패딩을 적용하여 모든 데이터를 동일한 길이로 맞춤
    for i in range(len(keypoints)):
        if keypoints[i].shape[0] < max_length:
            padding = np.zeros((max_length - keypoints[i].shape[0], keypoints[i].shape[1]))
            keypoints[i] = np.vstack((keypoints[i], padding))

    return np.array(keypoints), np.array(labels), max_length

# 키포인트 데이터 로드
keypoints_output_folder = 'keypoints'
X, y, max_length = load_keypoint_data(keypoints_output_folder)
X = np.nan_to_num(X)

# max_length 저장
np.save('max_length.npy', max_length)

# 데이터셋 분할 (정상 행동 데이터로만 구성됨)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 및 훈련
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 모델 저장
model_path = 'anomaly_detection_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
