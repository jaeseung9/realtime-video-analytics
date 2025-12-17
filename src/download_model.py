#!/usr/bin/env python
"""
ONNX 감정 인식 모델 다운로드 스크립트
"""

import os
import urllib.request
import sys


def download_model():
    """FERPlus 감정 인식 모델 다운로드"""
    
    # 모델 디렉토리 생성
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ {models_dir} 디렉토리 생성")
    
    # 모델 파일 경로
    model_path = os.path.join(models_dir, "emotion-ferplus.onnx")
    
    # 이미 존재하면 스킵
    if os.path.exists(model_path):
        print(f"✅ 모델이 이미 존재합니다: {model_path}")
        return
    
    # 다운로드 URL
    url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    
    print(f"📥 모델 다운로드 중...")
    print(f"   URL: {url}")
    print(f"   저장 위치: {model_path}")
    
    try:
        # 다운로드 진행률 표시
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f"\r   진행률: {percent:.1f}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, model_path, download_progress)
        print("\n✅ 모델 다운로드 완료!")
        
        # 파일 크기 확인
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   파일 크기: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print("\n대체 방법:")
        print("1. 수동 다운로드:")
        print(f"   wget {url} -O {model_path}")
        print("\n2. 또는 브라우저에서 다운로드 후 models/ 폴더에 복사")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
