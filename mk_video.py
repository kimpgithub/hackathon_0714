import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from PIL import Image

# 비디오 파일 경로와 그래프 이미지 경로 설정
video_path = 'test.avi'
graph_image_path = 'Figure_1.png'
output_video_path = 'output.mp4'

# 비디오 캡처 설정
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps

# 그래프 이미지 로드
graph_img = Image.open(graph_image_path)
graph_img = np.array(graph_img)

# Matplotlib 설정
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 첫 번째 서브플롯에 비디오 프레임 표시
ax1.set_title('Video')
video_frame = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# 두 번째 서브플롯에 그래프 이미지 표시
ax2.set_title('Graph')
ax2.imshow(graph_img)
vertical_line = ax2.axvline(x=0, color='r', linestyle='--')

# 그래프 초기화 함수
def init_graph():
    vertical_line.set_xdata([0])
    return video_frame, vertical_line

# 임시 디렉토리 생성
temp_dir = 'temp_frames'
os.makedirs(temp_dir, exist_ok=True)

# 프레임 업데이트 함수
def update(frame):
    # 비디오 프레임 읽기
    ret, video_img = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, video_img = cap.read()

    # 비디오 프레임 업데이트
    video_frame.set_data(cv2.cvtColor(video_img, cv2.COLOR_BGR2RGB))
    
    # 그래프 업데이트 (세로선 이동)
    vertical_line.set_xdata([frame * (graph_img.shape[1] / total_frames)])
    
    # 현재 프레임 저장
    plt.savefig(os.path.join(temp_dir, f'frame_{frame:05d}.png'))
    
    return video_frame, vertical_line

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=total_frames, init_func=init_graph, blit=True, interval=1000/fps)

plt.tight_layout()
plt.show()

# 비디오 캡처 해제
cap.release()

# 프레임을 비디오로 저장
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_height, frame_width, _ = cv2.imread(os.path.join(temp_dir, 'frame_00000.png')).shape
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

for frame_num in range(total_frames):
    frame_path = os.path.join(temp_dir, f'frame_{frame_num:05d}.png')
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()

# 임시 디렉토리 삭제
import shutil
shutil.rmtree(temp_dir)
