import cv2
import numpy as np
from typing import Tuple, Dict


class EmotionClassifier:
    """얼굴 감정 분석 (ONNX 모델 사용 예정, 현재는 더미)"""
    
    def __init__(self, model_path=None):
        # 감정 레이블
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # TODO: ONNX 모델 로드
        # 현재는 더미 구현
        self.model = None
        
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리"""
        # 48x48 그레이스케일로 변환
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        
        # 정규화
        face_normalized = face_resized.astype('float32') / 255.0
        
        # 배치 차원 추가
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=0)
        
        return face_input
    
    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """감정 예측"""
        # TODO: 실제 모델 추론
        # 현재는 더미 구현 (랜덤)
        
        # 얼굴 밝기 기반 더미 예측
        brightness = np.mean(face_img)
        
        if brightness > 150:
            emotion = "Happy"
            confidence = 0.85
        elif brightness > 100:
            emotion = "Neutral" 
            confidence = 0.75
        else:
            emotion = "Sad"
            confidence = 0.65
            
        return emotion, confidence
    
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """감정별 색상"""
        colors = {
            'Happy': (0, 255, 0),      # 초록
            'Sad': (255, 0, 0),        # 파랑
            'Angry': (0, 0, 255),      # 빨강
            'Surprise': (0, 255, 255), # 노랑
            'Fear': (255, 0, 255),     # 보라
            'Disgust': (128, 0, 128),  # 자주
            'Neutral': (128, 128, 128) # 회색
        }
        return colors.get(emotion, (255, 255, 255))
    
    def draw_emotion(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int],
                    emotion: str, confidence: float) -> np.ndarray:
        """프레임에 감정 표시"""
        x, y, w, h = face_rect
        color = self.get_emotion_color(emotion)
        
        # 감정 텍스트
        text = f"{emotion} ({confidence:.2f})"
        
        # 배경 박스
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 30), (x + text_size[0], y - 5), color, -1)
        
        # 텍스트
        cv2.putText(frame, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 얼굴 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        return frame