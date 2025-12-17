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
from failure_predictor import FailurePredictor

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
        failure_predictor = FailurePredictor(window_size=30)
        return face_detector, emotion_classifier, quality_analyzer, failure_predictor
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None, None

# 모델 로드
face_detector, emotion_classifier, quality_analyzer, failure_predictor = load_models()

# 세션 상태 초기화
if 'running' not in st.session_state:
    st.session_state.running = False
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'quality_history' not in st.session_state:
    st.session_state.quality_history = []
if 'failure_history' not in st.session_state:
    st.session_state.failure_history = []

# 사이드바 설정
st.sidebar.title("⚙️ 설정")

# 분석 옵션
st.sidebar.subheader("📊 분석 옵션")
enable_face = st.sidebar.checkbox("얼굴 검출", value=True)
enable_emotion = st.sidebar.checkbox("감정 분석", value=True)
enable_quality = st.sidebar.checkbox("품질 분석", value=True)
enable_prediction = st.sidebar.checkbox("이상/고장 예측", value=True)
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

# 예측 임계값
if enable_prediction:
    st.sidebar.subheader("🔮 예측 설정")
    prediction_window = st.sidebar.slider("예측 윈도우 (프레임)", 10, 60, 30)
    if failure_predictor:
        failure_predictor.window_size = prediction_window

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

# 예측 섹션 (새로 추가)
if enable_prediction:
    st.subheader("🚨 이상/고장 예측")
    pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 2])
    with pred_col1:
        prediction_gauge = st.empty()
    with pred_col2:
        prediction_status = st.empty()
    with pred_col3:
        prediction_reason = st.empty()

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
        st.session_state.failure_history = []

with control_col2:
    if st.button("⏹️ 정지"):
        st.session_state.running = False

with control_col3:
    if st.button("🗑️ 기록 초기화"):
        st.session_state.metrics_history = []
        st.session_state.quality_history = []
        st.session_state.failure_history = []

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
                
                # 예측기에 메트릭 추가
                if enable_prediction and failure_predictor:
                    failure_predictor.add_metrics(quality_results)
                
                # 품질 정보 오버레이
                if quality_results:
                    quality_text = f"Quality: {quality_results['quality_status']} ({quality_results['quality_score']:.2f})"
                    color = (0, 255, 0) if quality_results['quality_score'] > quality_threshold else (0, 0, 255)
                    cv2.putText(frame, quality_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 이상/고장 예측 (10프레임마다)
            if enable_prediction and failure_predictor and frame_count % 10 == 0:
                prob, status, reason = failure_predictor.predict_failure()
                st.session_state.failure_history.append(prob)
                
                # 예측 결과 표시
                with prediction_gauge:
                    # 확률을 퍼센트로 표시
                    st.metric("이상 확률", f"{prob*100:.1f}%")
                
                with prediction_status:
                    # 상태별 색상 이모지
                    status_emojis = {
                        "Normal": "🟢",
                        "Caution": "🟡", 
                        "Warning": "🟠",
                        "Critical": "🔴"
                    }
                    emoji = status_emojis.get(status, "⚪")
                    st.metric("상태", f"{emoji} {status}")
                
                with prediction_reason:
                    if reason != "정상":
                        st.warning(f"⚠️ 원인: {reason}")
                    else:
                        st.success("✅ 시스템 정상")
                
                # 프레임에도 표시
                if prob > 0.5:
                    alert_text = f"[{status}] {prob:.1%}"
                    alert_color = (0, 0, 255) if prob > 0.7 else (0, 165, 255)
                    cv2.putText(frame, alert_text, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            
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
                            # 감정 이모지 매핑
                            emotion_emojis = {
                                'Happy': '😊', 'Sad': '😢', 'Anger': '😠',
                                'Surprise': '😮', 'Fear': '😨', 'Disgust': '🤢',
                                'Neutral': '😐', 'Contempt': '😏'
                            }
                            emoji = emotion_emojis.get(data['emotion'], '🙂')
                            st.write(f"{emoji} Face {data['face_id']}: **{data['emotion']}** ({data['confidence']:.2%})")
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
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[2, 1])
                    
                    # 품질 점수 그래프
                    ax1.plot(st.session_state.quality_history[-100:], color='blue', linewidth=2, label='Quality')
                    ax1.axhline(y=quality_threshold, color='r', linestyle='--', label=f'Threshold: {quality_threshold}')
                    ax1.set_ylabel('Quality Score')
                    ax1.set_ylim([0, 1])
                    ax1.legend(loc='upper right')
                    ax1.grid(True, alpha=0.3)
                    
                    # 이상 확률 그래프
                    if st.session_state.failure_history:
                        ax2.plot(st.session_state.failure_history[-100:], color='red', linewidth=2)
                        ax2.fill_between(range(len(st.session_state.failure_history[-100:])), 
                                        st.session_state.failure_history[-100:], 
                                        alpha=0.3, color='red')
                        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
                        ax2.set_ylabel('Failure Prob')
                        ax2.set_xlabel('Frame')
                        ax2.set_ylim([0, 1])
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            # MLflow 로깅
            if analytics_manager and frame_count % 10 == 0:
                log_data = sys_metrics.copy()
                if quality_results:
                    log_data.update(quality_results)
                if enable_prediction and failure_predictor:
                    summary = failure_predictor.get_summary()
                    log_data['failure_probability'] = summary['failure_probability']
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
            
            # 예측 요약
            if st.session_state.failure_history:
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("평균 이상 확률", f"{np.mean(st.session_state.failure_history)*100:.1f}%")
                col2.metric("최대 이상 확률", f"{np.max(st.session_state.failure_history)*100:.1f}%")
                critical_count = sum(1 for p in st.session_state.failure_history if p > 0.7)
                col3.metric("Critical 횟수", critical_count)

# 하단 정보
st.divider()
st.info("💡 Docker + MLflow + ONNX 기반 실시간 영상 분석(감정/품질/예측) 통합 시스템")

# 디버그 정보
with st.expander("🔧 시스템 정보"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**모델 상태:**")
        st.write(f"- 얼굴 검출: {'✅' if face_detector else '❌'}")
        st.write(f"- 감정 분석: {'✅' if emotion_classifier else '❌'}")
        if emotion_classifier and hasattr(emotion_classifier, 'session'):
            st.write(f"  - ONNX 모델: {'✅ 로드됨' if emotion_classifier.session else '⚠️ 더미모드'}")
        st.write(f"- 품질 분석: {'✅' if quality_analyzer else '❌'}")
        st.write(f"- 이상 예측: {'✅' if failure_predictor else '❌'}")
    with col2:
        st.write("**데이터 버퍼:**")
        st.write(f"- 품질 기록: {len(st.session_state.quality_history)} 프레임")
        st.write(f"- 예측 기록: {len(st.session_state.failure_history)} 프레임")
        if failure_predictor:
            summary = failure_predictor.get_summary()
            st.write(f"- 예측 버퍼: {summary['buffer_size']}/{failure_predictor.window_size} 프레임")

# 임시 파일 정리
if temp_file_path and os.path.exists(temp_file_path):
    try:
        os.unlink(temp_file_path)
    except:
        pass
