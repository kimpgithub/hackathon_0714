import os
import pandas as pd
import yt_dlp as youtube_dl

# 데이터 다운로드 함수
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
