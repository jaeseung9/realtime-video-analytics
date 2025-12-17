import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier
from quality_analyzer import QualityAnalyzer
from metrics_collector import MetricsCollector
from analytics_manager import AnalyticsManager

# 페이지 설정
st.set_page_config(
    page_title="Real-time Video Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📹 Real-time Video Quality Analysis & Face Emotion Detection")

# 초기화 함수들
@st.cache_resource
def load_models():
    """모든 모델과 분석기 로드"""
    try:
        face_detector = FaceDetector()
        emotion_classifier = EmotionClassifier()
        quality_analyzer = QualityAnalyzer()
        return face_detector, emotion_classifier, quality_analyzer
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None

# 모델 로드
face_detector, emotion_classifier, quality_analyzer = load_models()

# 세션 상태 초기화
if 'running' not in st.session_state:
    st.session_state.running = False
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'quality_history' not in st.session_state:
    st.session_state.quality_history = []

# 사이드바 설정
st.sidebar.title("⚙️ 설정")

# 분석 옵션
st.sidebar.subheader("📊 분석 옵션")
enable_face = st.sidebar.checkbox("얼굴 검출", value=True)
enable_emotion = st.sidebar.checkbox("감정 분석", value=True)
enable_quality = st.sidebar.checkbox("품질 분석", value=True)
enable_mlflow = st.sidebar.checkbox("MLflow 로깅", value=False)

# 비디오 소스
st.sidebar.subheader("📹 비디오 소스")
source_type = st.sidebar.radio("소스 선택", ["웹캠", "영상 파일", "유튜브 링크"])

video_source = None
temp_file_path = None

if source_type == "웹캠":
    video_source = 0
    st.sidebar.info("💡 웹캠 사용 중")
elif source_type == "영상 파일":
    uploaded_file = st.sidebar.file_uploader(
        "영상 업로드", 
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
        video_source = temp_file_path
        st.sidebar.success(f"✅ {uploaded_file.name}")
elif source_type == "유튜브 링크":
    youtube_url = st.sidebar.text_input(
        "유튜브 URL 입력",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if youtube_url:
        st.sidebar.info("💡 유튜브 영상 로드 중...")
        try:
            import yt_dlp
            
            ydl_opts = {
                "quiet": True,
                "format": "best[ext=mp4]/best"
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                video_source = info["url"]
                
            st.sidebar.success("✅ 유튜브 영상 로드 완료")
        except ImportError:
            st.sidebar.error("❌ yt-dlp 설치 필요: pip install yt-dlp")
        except Exception as e:
            st.sidebar.error(f"❌ 유튜브 로드 실패: {e}")

# FPS 제한
fps_limit = st.sidebar.slider("FPS 제한", 1, 30, 15)

# 품질 임계값
quality_threshold = st.sidebar.slider("품질 임계값", 0.0, 1.0, 0.6, 0.1)

# 메인 레이아웃
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("🎥 비디오 스트림")
    video_placeholder = st.empty()

with col2:
    st.subheader("📈 실시간 메트릭")
    metrics_placeholder = st.empty()

with col3:
    st.subheader("💭 감정 분석")
    emotion_placeholder = st.empty()

# 품질 그래프
st.subheader("📊 품질 추이")
quality_chart_placeholder = st.empty()

# 시스템 메트릭
system_metrics_placeholder = st.empty()

# 컨트롤 버튼
control_col1, control_col2, control_col3 = st.columns([1, 1, 4])
with control_col1:
    if st.button("▶️ 시작", type="primary"):
        st.session_state.running = True
        st.session_state.metrics_history = []
        st.session_state.quality_history = []

with control_col2:
    if st.button("⏹️ 정지"):
        st.session_state.running = False

with control_col3:
    if st.button("🗑️ 기록 초기화"):
        st.session_state.metrics_history = []
        st.session_state.quality_history = []

# 메인 처리 루프
if st.session_state.running and video_source is not None:
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error("❌ 비디오를 열 수 없습니다")
        st.session_state.running = False
    else:
        # 매니저 초기화
        metrics_collector = MetricsCollector()
        analytics_manager = AnalyticsManager() if enable_mlflow else None
        
        # MLflow 실행 시작
        if analytics_manager:
            analytics_manager.start_run()
        
        frame_count = 0
        skip_frames = max(1, 30 // fps_limit)  # 프레임 스킵 계산
        analysis_interval = 5  # 5프레임마다 분석
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.info("🎬 영상 종료")
                break
            
            # 프레임 스킵 (FPS 조절)
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # 시스템 메트릭은 항상 수집
            sys_metrics = metrics_collector.get_all_metrics()
            
            # 무거운 분석은 일정 간격으로만
            should_analyze = (frame_count % analysis_interval == 0)
            
            # 품질 분석 (간격마다)
            quality_results = None
            if enable_quality and quality_analyzer and should_analyze:
                quality_results = quality_analyzer.analyze_frame(frame)
                st.session_state.quality_history.append(quality_results['quality_score'])
                
                # 품질 정보 오버레이
                if quality_results:
                    quality_text = f"Quality: {quality_results['quality_status']} ({quality_results['quality_score']:.2f})"
                    color = (0, 255, 0) if quality_results['quality_score'] > quality_threshold else (0, 0, 255)
                    cv2.putText(frame, quality_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 얼굴 검출 및 감정 분석 (간격마다)
            if enable_face and face_detector and should_analyze:
                faces = face_detector.detect_faces(frame)
                
                if enable_emotion and emotion_classifier and len(faces) > 0:
                    emotions_data = []
                    for i, (x, y, w, h) in enumerate(faces):
                        # 얼굴 영역 추출
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # 감정 예측
                        emotion, confidence = emotion_classifier.predict_emotion(face_roi)
                        emotions_data.append({
                            'face_id': i,
                            'emotion': emotion,
                            'confidence': confidence
                        })
                        
                        # 프레임에 그리기
                        frame = emotion_classifier.draw_emotion(
                            frame, (x, y, w, h), emotion, confidence
                        )
                        
                        # MLflow 로깅
                        if analytics_manager:
                            analytics_manager.log_emotion(emotion, confidence, i)
                    
                    # 감정 표시
                    with emotion_placeholder.container():
                        for data in emotions_data:
                            st.write(f"Face {data['face_id']}: **{data['emotion']}** ({data['confidence']:.2%})")
                else:
                    # 얼굴만 그리기
                    frame = face_detector.draw_faces(frame, faces)
            
            # FPS 정보 추가
            cv2.putText(frame, f"FPS: {sys_metrics['current_fps']:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 비디오 표시
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # 메트릭 표시
            with metrics_placeholder.container():
                st.metric("FPS", f"{sys_metrics['current_fps']:.1f}")
                st.metric("CPU", f"{sys_metrics['cpu_percent']:.1f}%")
                st.metric("Memory", f"{sys_metrics['memory_percent']:.1f}%")
                if quality_results:
                    st.metric("품질 점수", f"{quality_results['quality_score']:.2f}")
            
            # 품질 그래프 업데이트 (30프레임마다)
            if len(st.session_state.quality_history) > 0 and frame_count % 30 == 0:
                with quality_chart_placeholder.container():
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(st.session_state.quality_history[-100:], color='blue', linewidth=2)
                    ax.axhline(y=quality_threshold, color='r', linestyle='--', label=f'Threshold: {quality_threshold}')
                    ax.set_ylabel('Quality Score')
                    ax.set_xlabel('Frame')
                    ax.set_ylim([0, 1])
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            # MLflow 로깅
            if analytics_manager and frame_count % 10 == 0:
                log_data = sys_metrics.copy()
                if quality_results:
                    log_data.update(quality_results)
                analytics_manager.log_metrics(log_data, step=frame_count)
            
            frame_count += 1
        
        # 정리
        cap.release()
        if analytics_manager:
            analytics_manager.end_run()
        
        # 최종 통계
        with system_metrics_placeholder.container():
            st.success("✅ 처리 완료!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 프레임", frame_count)
            col2.metric("평균 FPS", f"{sys_metrics.get('avg_fps', 0):.1f}")
            if st.session_state.quality_history:
                col3.metric("평균 품질", f"{np.mean(st.session_state.quality_history):.2f}")
                col4.metric("최저 품질", f"{np.min(st.session_state.quality_history):.2f}")

# 하단 정보
st.divider()
st.info("💡 Docker + MLflow + PyTorch 기반 실시간 영상 분석(감정/품질/OCR/예측) 통합 시스템")

# 임시 파일 정리
if temp_file_path and os.path.exists(temp_file_path):
    try:
        os.unlink(temp_file_path)
    except:
        pass