# 📹 Real-time Video Quality Analysis & Face Emotion Detection System

> **프로젝트명**: realtime-video-analyticsQuality  
> **목적**: 실시간 영상에서 **(1) 얼굴 검출/감정 표시**와 **(2) CCTV 품질 지표(블러/밝기/노이즈/프리즈) 모니터링**을 동시에 수행하고,  
> **MLflow + CSV 로그**로 **시계열 추적 가능한 분석 환경**을 만드는 **통합 데모(MVP)** 입니다.

- **테스트 환경**: 노트북 웹캠 / 로컬 영상 파일 / (옵션) 유튜브 스트림
- **타깃 도메인**: 영상 분석(예: CCTV) / CV

---

## ✨ 핵심 가치(이 프로젝트가 “좋은 이유”)

- **실시간 파이프라인을 한 화면에 통합**: 영상 스트림 + 품질 지표 + 감정 표시 + 시스템 메트릭(FPS/CPU/Mem)
- **실험/로그 기반 개발 흐름**: MLflow 로깅 + CSV 시계열 저장으로 “데모 → 비교/분석”이 가능
- **현업 친화 품질 지표**: Blur(Laplacian), Brightness(mean), Noise(고주파 proxy), Freeze(프레임 변화량)

> ⚠️ **중요(정직한 현재 상태)**  
> 현재 레포의 `EmotionClassifier` / `OCRPipeline`은 **더미(placeholder) 구현**이며,  
> ONNX 감정모델 / EasyOCR / RandomForest 예측은 **다음 단계(로드맵)**로 설계되어 있습니다.

---

## 📌 현재 구현 범위 (MVP)

### 1) Streamlit 대시보드 (`app.py`)

- **비디오 소스**: 웹캠 / 파일 업로드 / (옵션) 유튜브 URL(yt-dlp 필요)
- **실시간 표시**
  - 좌: 비디오 프레임(오버레이: Quality/FPS/감정)
  - 중: FPS/CPU/Memory/품질점수
  - 우: 얼굴별 감정 텍스트 출력
  - 하단: **Quality Score 시계열 그래프**

### 2) 품질 분석 (`quality_analyzer.py`)

- `blur_score` : Laplacian variance 기반, **0~1 정규화(1이 가장 블러)**
- `brightness` : gray 평균 (0~255)
- `noise_level` : 고주파 에너지 proxy (0~1)
- `is_frozen` : 연속 프레임 평균차가 작으면 카운트, 임계치 이상이면 True
- `quality_score` : 위 지표를 가중 결합하여 0~1 점수 산출

### 3) 얼굴 검출 (`face_detector.py`)

- Haar Cascade + `detectMultiScale3`(가능 시) + **eye 검증(require_eye)**로 오검출 억제
- Windows 한글 경로 문제 우회: `C:\opencv_temp`로 XML 복사 후 로드

### 4) 시스템 메트릭 (`metrics_collector.py`)

- `psutil` 기반 CPU/Mem + FPS/Latency 계산(최근 30프레임 이동평균)

### 5) MLflow/CSV 로깅 (`analytics_manager.py`)

- MLflow: numeric metric 로깅(기본 `MLFLOW_TRACKING_URI=http://localhost:5000`)
- CSV: `logs/metrics.csv`, `logs/emotions.csv`로 누적 저장(버퍼 10개 단위 flush)

---

## 🛠️ 기술 스택

### 현재 코드에서 실제 사용

- Python 3.11
- Streamlit
- OpenCV
- NumPy / Pandas
- Matplotlib
- MLflow
- psutil
- (옵션) `yt-dlp` (유튜브 입력 모드 사용 시)

### 다음 단계(로드맵에 포함)

- ONNX Runtime (emotion-ferplus.onnx 등)
- EasyOCR (텍스트 + confidence)
- Scikit-learn(RandomForest) 기반 이상 예측
- (선택) PyTorch 기반 품질 스코어러

---

## 📂 프로젝트 구조(현재 src.zip 기준)

```
src/
  app.py
  analytics_manager.py
  face_detector.py
  emotion_classifier.py
  quality_analyzer.py
  metrics_collector.py
  failure_predictor.py
  ocr_pipeline.py
```

> 레포 형태로 정리할 때는 아래 구조로 확장 예정입니다.

```
face-emotion-analytics/
  Dockerfile
  docker-compose.yml
  requirements.txt
  resources/
    haarcascade_frontalface_default.xml
    haarcascade_eye_tree_eyeglasses.xml
  logs/
    metrics.csv
    emotions.csv
```

---

## 🚀 실행 방법

### 1) 로컬 실행 (가장 빠른 데모)

```bash
# (권장) 가상환경
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install streamlit opencv-python numpy pandas matplotlib mlflow psutil
# 유튜브 입력 모드 쓰면:
pip install yt-dlp

streamlit run app.py  # app.py가 현재 경로에 있을 때
# 또는 레포 구조가 src/app.py라면:
# streamlit run src/app.py
```

- 접속: `http://localhost:8501`

### 2) MLflow UI 같이 보기 (선택)

```bash
# 로컬에서 MLflow 서버/UI 실행
mlflow ui --host 0.0.0.0 --port 5000
```

- 접속: `http://localhost:5000`
- Streamlit 사이드바에서 **“MLflow 로깅”** 체크 후 실행

---

## ⚙️ 설정값(주요 파라미터)

Streamlit 사이드바에서 실시간 조절:

- FPS 제한
- 품질 임계값(quality_threshold)
- 얼굴 검출 / 감정 분석 / 품질 분석 ON/OFF
- MLflow 로깅 ON/OFF

환경변수(선택):

- `MLFLOW_TRACKING_URI` (기본: `http://localhost:5000`)
- `VIDEO_SOURCE` (AnalyticsManager 파라미터 기록용)
- `QUALITY_THRESHOLD` (AnalyticsManager 파라미터 기록용)

---

## 🧪 로그/지표 확인

### CSV 로그

- `logs/metrics.csv` : 시스템 메트릭 + (옵션) 품질 메트릭
- `logs/emotions.csv` : face_id, emotion, confidence 시계열

### MLflow

- FPS/CPU/Mem/Latency + 품질 지표를 실험 단위로 비교 가능

---

## ✅ 프로젝트 불량 보고 (Known Issues / Tech Debt)

이력서/면접에서 “다음 단계까지 고민한 사람”으로 보이게 만드는 포인트입니다.

### 기능 미완(의도된 placeholder)

- `EmotionClassifier`: 현재는 **밝기 기반 더미 예측** → ONNX 추론으로 교체 필요
- `OCRPipeline`: 현재는 **텍스트 밀도 더미** → EasyOCR 기반 text+confidence로 교체 필요
- `FailurePredictor`: 규칙 기반(내장) 구현은 있으나 **Streamlit UI 미연동**
- Docker/Compose/모델 파일/테스트 폴더는 **레포 정리 단계에서 추가 예정**

### 실행/성능 이슈

- Streamlit에서 while-loop 기반 실시간 처리 → 장시간 실행 시 리소스 사용 증가 가능  
  (개선안: thread/async 분리, `st.session_state` 기반 제어 강화, frame queue 적용)
- Haar Cascade 파일은 `resources/`에 필요
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_eye_tree_eyeglasses.xml`

---

## 🗺️ 앞으로 할 것 (Roadmap)

### 1) 감정 모델 ONNX 적용 (가장 임팩트 큼)

- `emotion-ferplus.onnx` + onnxruntime로 실제 추론
- 얼굴 ROI 전처리(48x48, normalize) 및 결과 softmax/confidence 적용

### 2) OCR(EasyOCR) 시계열 통합

- 텍스트/Confidence를 `metrics.csv`에 함께 저장
- “품질 저하 ↔ OCR 신뢰도 하락” 상관관계 관측 UI 추가

### 3) 이상/고장 예측(Scikit-learn)

- 슬라이딩 윈도우 feature(mean/std/slope) → RandomForest 확률 출력
- Streamlit에 게이지/라인차트로 표시 + MLflow에 ROC 등 실험 기록

### 4) Docker Compose 패키징

- Streamlit + MLflow를 한 번에 실행
- volume 마운트로 `mlruns/` 영속화

---

## 🏁 성과(이력서용 요약)

- OpenCV 기반 **실시간 영상 분석 파이프라인**(입력→분석→시각화→로깅) 구축
- Blur/Brightness/Noise/Freeze 기반 **품질 스코어링** 설계 및 시계열 모니터링 구현
- psutil 기반 **시스템 메트릭(FPS/CPU/Mem) 계측**으로 성능 관측 가능한 대시보드 제공
- MLflow + CSV로 “실험/로그 중심” 개발 구조를 적용하여 비교 가능한 분석 환경 구성

---

## 라이선스

학습/포트폴리오 목적 (추후 공개 시 LICENSE 추가 예정)
