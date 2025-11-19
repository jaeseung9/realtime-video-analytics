# 🎭 Real-time Face Emotion Analytics Dashboard

> 웹캠 영상에서 얼굴을 실시간 검출하고 감정을 분석하여 대시보드 형태로 시각화하는 AI 웹 애플리케이션

-- 현재 개발 진행 중 --

---

## 📑 목차

1. [📌 프로젝트 개요](#-프로젝트-개요)
2. [🎯 핵심 기능](#-핵심-기능)
3. [🛠️ 기술 스택](#️-기술-스택)
4. [📂 프로젝트 구조](#-프로젝트-구조)
5. [🚀 주요 구현 사항](#-주요-구현-사항)
6. [🎨 UI 화면 구성](#-ui-화면-구성-streamlit)
7. [⚡ 성능 최적화](#-성능-최적화)
8. [📊 감정 분석 모델 정보](#-감정-분석-모델-정보)
9. [🎯 기술적 성과](#-기술적-성과)
10. [📝 개발 일정](#-개발-일정-1주)
11. [🎬 시연 영상](#-시연-영상)
12. [📦 설치 및 실행](#-설치-및-실행)
13. [📋 Requirements](#-requirements)
14. [🔮 향후 개선 계획](#-향후-개선-계획)
15. [👨‍💻 개발자](#-개발자)
16. [📄 라이선스](#-라이선스)
17. [🙏 참고 자료](#-참고-자료)

---

## 📌 프로젝트 개요

실시간 웹캠 영상에서 얼굴을 검출하고, 딥러닝 모델을 활용하여 감정을 분석한 뒤, 
Streamlit 기반 대시보드로 시각화하는 컴퓨터 비전 프로젝트입니다.

**개발 기간**: 1주  
**주요 목표**: 딥러닝 + 컴퓨터비전 + 실시간 스트림 + UI 종합 역량 어필

---

## 🎯 핵심 기능

### 1. **실시간 얼굴 감정 인식**
- OpenCV를 활용한 실시간 얼굴 검출 (Haar Cascade)
- ONNX 기반 감정 분류 모델 (FER2013 기반)
- 8가지 감정 분류: Neutral, Happy, Sad, Angry, Surprise, Fear, Disgust, Contempt

### 2. **감정 분석 대시보드**
- 현재 감정 및 확률 실시간 표시
- 감정별 확률 바 그래프
- 시간에 따른 감정 변화 추세 그래프 (Line Chart)
- 감정 누적 비율 분석 (Pie Chart)

### 3. **데이터 로깅 및 분석**
- 감정 변화 이력 자동 기록
- CSV 형태로 로그 저장 및 다운로드 기능
- 감정별 통계 데이터 제공

---

## 🛠️ 기술 스택

### Core Technologies
- **Python 3.10**: 메인 개발 언어
- **OpenCV 4.x**: 실시간 영상 처리 및 얼굴 검출
- **ONNX Runtime**: 경량화된 딥러닝 추론
- **Streamlit**: 웹 기반 대시보드 UI

### Libraries
- **NumPy**: 배열 연산 및 데이터 처리
- **Pandas**: 데이터 분석 및 로그 관리
- **Matplotlib / Plotly**: 데이터 시각화

---

## 📂 프로젝트 구조

```
FaceEmotionAnalytics/
├── models/
│   └── emotion-ferplus.onnx          # 감정 인식 ONNX 모델
├── src/
│   ├── face_detector.py              # 얼굴 검출 모듈
│   ├── emotion_classifier.py         # 감정 분류 모듈
│   ├── analytics_manager.py          # 분석 데이터 관리
│   ├── utils.py                      # 유틸리티 함수
│   └── app.py                        # Streamlit 메인 앱
├── resources/
│   └── haarcascade_frontalface_default.xml  # Haar Cascade 파일
├── logs/
│   └── emotions.csv                  # 감정 로그 데이터
├── README.md
└── requirements.txt
```

---

## 🚀 주요 구현 사항

### 1. **FaceDetector** - 얼굴 검출
```python
class FaceDetector:
    def __init__(self):
        # Haar Cascade 로드
        self.face_cascade = cv2.CascadeClassifier(
            'resources/haarcascade_frontalface_default.xml'
        )
    
    def detect(self, frame):
        # 그레이스케일 변환 후 얼굴 영역(ROI) 추출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
```

### 2. **EmotionClassifier** - 감정 분류
```python
class EmotionClassifier:
    def __init__(self, model_path):
        # ONNX 모델 로드
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.labels = ['Neutral', 'Happy', 'Sad', 'Angry', 
                      'Surprise', 'Fear', 'Disgust', 'Contempt']
    
    def predict(self, face_roi):
        # 전처리 → 추론 → Softmax
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (64, 64))
        self.net.setInput(blob)
        prob = self.net.forward()
        confidence = np.max(prob)
        label = self.labels[np.argmax(prob)]
        return label, float(confidence)
```

### 3. **AnalyticsManager** - 데이터 분석
```python
class AnalyticsManager:
    def __init__(self):
        self.history = []
        self.counts = {label: 0 for label in range(8)}
    
    def update(self, label, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append((timestamp, label, confidence))
```

---

## 🎨 UI 화면 구성 (Streamlit)

### 레이아웃
- **왼쪽**: 웹캠 실시간 영상 스트리밍
- **오른쪽 상단**: 
  - 현재 감정 표시 (큰 텍스트)
  - 감정 확률 바 그래프
- **오른쪽 중단**: 
  - 실시간 감정 변화 추세 그래프 (Line Chart)
  - 감정 누적 비율 (Pie Chart)
- **하단**: 
  - 감정 로그 다운로드 버튼
  - 통계 데이터 테이블

---

## ⚡ 성능 최적화

1. **해상도 다운스케일링**: 720p → 480p 변환으로 FPS 개선
2. **ROI 단위 추론**: 얼굴 영역만 모델에 입력하여 연산량 최소화
3. **Streamlit 최적화**: `st.empty()` 반복 렌더링으로 화면 갱신 효율화
4. **History 제한**: 최근 300프레임으로 메모리 사용량 제한

---

## 📊 감정 분석 모델 정보

| 항목 | 설명 |
|------|------|
| **모델 아키텍처** | FER2013 기반 CNN |
| **입력 크기** | 64×64 Grayscale |
| **출력** | 8-class Softmax |
| **정확도** | ~75% (FER2013 데이터셋 기준) |
| **추론 속도** | ~30 FPS (CPU) |

---

## 🎯 기술적 성과

✅ **실시간 CV + 딥러닝 모델 결합 경험**  
✅ **웹 기반 인터랙티브 대시보드 구현**  
✅ **OpenCV 전처리 → ONNX 추론 파이프라인 구축**  
✅ **데이터 시각화 및 분석 기능 구현**

---

## 📝 개발 일정 (1주)

| Day | 작업 내용 |
|-----|----------|
| **Day 1** | 환경 구성, 프로젝트 세팅, 모델 다운로드 |
| **Day 2** | 얼굴 검출 기능 구현 (Haar Cascade) |
| **Day 3** | 감정 인식 모델 연동 및 테스트 |
| **Day 4** | 실시간 스트리밍 + 감정분석 통합 |
| **Day 5** | 대시보드 분석 기능 추가 (그래프, 통계) |
| **Day 6** | UI/UX 개선 및 예외 처리 |
| **Day 7** | 문서화, 시연 영상 촬영, 포트폴리오 정리 |

---

## 🎬 시연 영상

추가 예정

---

## 📦 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/face-emotion-analytics.git
cd face-emotion-analytics
```

### 2. 가상환경 생성 및 패키지 설치
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 애플리케이션 실행
```bash
streamlit run src/app.py
```

### 4. 브라우저 접속
자동으로 브라우저가 열리며, `http://localhost:8501`에서 확인 가능합니다.

---

## 📋 Requirements

```txt
opencv-python==4.8.0
streamlit==1.28.0
onnxruntime==1.16.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
plotly==5.17.0
```

---

## 🔮 향후 개선 계획

- [ ] 다중 얼굴 동시 분석 기능
- [ ] 더 정확한 감정 인식 모델 (Transformer 기반)
- [ ] 클라우드 배포 (AWS, GCP)
- [ ] 모바일 앱 연동
- [ ] 감정 데이터 기반 추천 시스템

---

## 👨‍💻 개발자

**[Your Name]**  
📧 Email: your.email@example.com  
🔗 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com)  
💼 Portfolio: [yourportfolio.com](https://yourportfolio.com)

---

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 참고 자료

- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [ONNX Runtime](https://onnxruntime.ai/)

---

<div align="center">
  <strong>⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐</strong>
</div>
