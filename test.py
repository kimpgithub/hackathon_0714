import os
import pandas as pd
import yt_dlp as youtube_dl
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. 데이터 다운로드
def download_video(url, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# 엑셀 파일 로드
file_path = 'YouTube_Link_InfAct.xlsx'
excel_data = pd.read_excel(file_path)

# 비디오 다운로드
output_folder = 'videos'
os.makedirs(output_folder, exist_ok=True)
for index, row in excel_data.iterrows():
    url = row['YouTube Link']
    if pd.notna(url):
        download_video(url, output_folder)

# 2. 키포인트 추출 (YOLOv8 사용)
model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델 로드

def extract_keypoints_from_video(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # YOLOv8을 사용하여 포즈 예측
        keypoints = results[0].keypoints.numpy()  # 예측된 포즈 키포인트를 numpy 배열로 변환
        np.save(os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_count:05d}.npy"), keypoints)
        
        frame_count += 1
    cap.release()

video_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.mp4')]
keypoints_output_folder = 'keypoints'
for video_file in video_files:
    extract_keypoints_from_video(video_file, keypoints_output_folder)

# 3. 데이터 로드 및 전처리
def load_keypoint_data(folder_path):
    keypoints = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder_path, file))
            keypoints.append(data)
            labels.append(0)  # 정상 행동 레이블
    return np.array(keypoints), np.array(labels)

X, y = load_keypoint_data(keypoints_output_folder)
X = np.nan_to_num(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 정의 및 훈련
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 5. 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
