import os
import fitz  # PyMuPDF
import yt_dlp as youtube_dl

# PDF 파일 경로
pdf_path = 'url-list.pdf'

# 비디오 저장 폴더 경로
output_folder = 'test_video'
os.makedirs(output_folder, exist_ok=True)

# PDF 파일에서 URL 추출
def extract_urls_from_pdf(pdf_path):
    urls = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text("text")
        lines = text.split('\n')
        for line in lines:
            if "http" in line:
                urls.append(line.strip().split()[-1])
    return urls

# 비디오 다운로드 함수
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

# PDF 파일에서 URL 추출
video_urls = extract_urls_from_pdf(pdf_path)

# 비디오 다운로드
for url in video_urls:
    download_video(url, output_folder)
