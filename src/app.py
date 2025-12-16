import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from face_detector import FaceDetector

st.set_page_config(page_title="Video + Face Detection", layout="wide")
st.title("📹 비디오 소스 + 얼굴 검출 테스트")

# 얼굴 검출기 초기화
@st.cache_resource
def load_face_detector():
    """얼굴 검출기 로드 (캐싱)"""
    try:
        detector = FaceDetector()
        return detector
    except Exception as e:
        st.error(f"얼굴 검출기 로드 실패: {e}")
        return None

face_detector = load_face_detector()

# 사이드바 설정
st.sidebar.title("⚙️ 설정")

# 얼굴 검출 ON/OFF
enable_face_detection = st.sidebar.checkbox("얼굴 검출 활성화", value=True)

# 검출 감도 조절
if enable_face_detection:
    detection_sensitivity = st.sidebar.slider(
        "검출 감도 (낮을수록 민감)", 
        min_value=2, 
        max_value=8, 
        value=3,
        help="값이 낮을수록 더 많은 얼굴을 검출하지만 오검출도 증가합니다"
    )

video_source_type = st.sidebar.radio(
    "비디오 소스 선택",
    ["웹캠", "영상 업로드", "유튜브 링크 (실험용)"]
)

video_source = None
temp_file_path = None

if video_source_type == "웹캠":
    st.sidebar.info("💡 웹캠을 사용합니다 (카메라 번호: 0)")
    video_source = 0

elif video_source_type == "영상 업로드":
    uploaded_file = st.sidebar.file_uploader(
        "영상 파일 업로드",
        type=["mp4", "avi", "mov", "mkv", "webm"]
    )

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
        video_source = temp_file_path
        st.sidebar.success(f"✅ 업로드 완료: {uploaded_file.name}")
    else:
        st.sidebar.warning("영상 파일을 업로드해주세요")

elif video_source_type.startswith("유튜브"):
    youtube_url = st.sidebar.text_input(
        "유튜브 URL 입력",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if youtube_url:
        st.sidebar.info("💡 유튜브 영상 정보를 불러오는 중...")
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
        except Exception as e:
            st.sidebar.error(f"❌ 유튜브 영상 로드 실패: {e}")
            st.sidebar.warning("yt-dlp 설치 필요: pip install yt-dlp")
    else:
        st.sidebar.warning("유튜브 URL을 입력해주세요")

fps_limit = st.sidebar.slider("FPS 제한", 1, 60, 30)

st.sidebar.divider()
st.sidebar.markdown("**현재 설정:**")
st.sidebar.write(f"- 소스: {video_source_type}")
st.sidebar.write(f"- 얼굴 검출: {'ON' if enable_face_detection else 'OFF'}")
st.sidebar.write(f"- FPS 제한: {fps_limit}")

# 컨트롤 버튼
col1, col2, _ = st.columns([1, 1, 4])
with col1:
    start = st.button("▶️ 시작")
with col2:
    stop = st.button("⏹️ 정지")

# 표시 영역
video_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()

# 상태 관리
if "running" not in st.session_state:
    st.session_state.running = False

if start and video_source is not None:
    st.session_state.running = True

if stop:
    st.session_state.running = False

# 비디오 처리
if st.session_state.running and video_source is not None:
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        st.error(f"❌ 비디오를 열 수 없습니다: {video_source}")
        st.session_state.running = False
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with status_placeholder.container():
            st.success("✅ 비디오 로드 완료")
            st.write(f"- 해상도: {width} x {height}")
            st.write(f"- FPS: {fps:.1f}")

        frame_count = 0
        total_faces = 0
        start_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.info("🎬 영상 종료")
                break

            # 얼굴 검출
            faces = []
            if enable_face_detection and face_detector is not None:
                faces = face_detector.detect_faces(frame, min_neighbors=detection_sensitivity)
                total_faces += len(faces)
                
                # 얼굴 그리기
                if len(faces) > 0:
                    frame = face_detector.draw_faces(frame, faces)

            # BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # FPS 계산
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # 정보 표시
            cv2.putText(frame_rgb, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"FPS: {current_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"Faces: {len(faces)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 화면에 표시
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # 메트릭 업데이트
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("현재 프레임", frame_count)
                col2.metric("현재 FPS", f"{current_fps:.1f}")
                col3.metric("검출된 얼굴", len(faces))

            frame_count += 1
            time.sleep(1 / fps_limit)

        cap.release()
        st.session_state.running = False
        
        # 최종 통계
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_faces = total_faces / frame_count if frame_count > 0 else 0
        
        with status_placeholder.container():
            st.success("✅ 처리 완료!")
            st.write(f"- 처리된 프레임: {frame_count}")
            st.write(f"- 처리 시간: {total_time:.2f}초")
            st.write(f"- 평균 FPS: {avg_fps:.2f}")
            st.write(f"- 평균 얼굴 수: {avg_faces:.2f}")

# 임시 파일 정리
if temp_file_path and os.path.exists(temp_file_path):
    try:
        os.unlink(temp_file_path)
    except:
        pass

st.divider()
st.info("💡 Step 2: 비디오에서 얼굴을 검출합니다.")