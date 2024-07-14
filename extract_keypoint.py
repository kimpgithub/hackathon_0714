import os
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 포즈 모델 로드
model = YOLO('yolov8n-pose.pt')

# 키포인트 추출 함수
def extract_keypoints_from_video(video_path, output_folder, frame_interval=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    # 비디오의 FPS 확인
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 설정된 프레임 간격에 따라 프레임 건너뛰기
        if frame_count % frame_interval == 0:
            results = model(frame)  # YOLOv8을 사용하여 포즈 예측

            # keypoints 확인 및 단순화
            keypoints = results[0].keypoints
            if keypoints is not None:
                keypoints = keypoints.xy.cpu().numpy()  # xy 좌표를 numpy 배열로 변환
                # keypoints의 구조 확인
                print(f"Frame {frame_count}: keypoints shape {keypoints.shape}")

                # keypoints를 저장 가능한 형태로 변환 (예: 2D 배열로 변환)
                keypoints = keypoints.reshape(-1, keypoints.shape[-1])
                np.save(os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{saved_frame_count:05d}.npy"), keypoints)
                saved_frame_count += 1
        
        frame_count += 1
    cap.release()

# 비디오 파일 경로
video_folder = 'videos'
keypoints_output_folder = 'keypoints'
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# 프레임 간격 설정 (예: 1초에 한 번 처리하려면 FPS만큼 건너뜀)
frame_interval = 30  # 예시: FPS가 30인 경우 1초에 한 번 처리

for video_file in video_files:
    extract_keypoints_from_video(video_file, keypoints_output_folder, frame_interval=frame_interval)
