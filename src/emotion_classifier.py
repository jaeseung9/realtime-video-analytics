import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import os


class EmotionClassifier:
    """ONNX Runtime 기반 얼굴 감정 분석"""
    
    def __init__(self, model_path: Optional[str] = None):
        # 감정 레이블 (FERPlus 순서)
        self.emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 
                        'Anger', 'Disgust', 'Fear', 'Contempt']
        
        # 모델 경로 설정
        if model_path is None:
            # 프로젝트 루트 기준 기본 경로
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "emotion-ferplus.onnx")
        
        self.model_path = model_path
        self.session = None
        
        # ONNX Runtime 로드 시도
        try:
            import onnxruntime as ort
            if os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                print(f"✅ ONNX 감정 모델 로드 성공: {self.model_path}")
            else:
                print(f"⚠️ ONNX 모델 파일 없음: {self.model_path}")
                print("➡️ 더미 모드로 동작합니다. 모델 다운로드 방법:")
                print("   wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx -O models/emotion-ferplus.onnx")
        except ImportError:
            print("⚠️ onnxruntime이 설치되지 않음. pip install onnxruntime")
        except Exception as e:
            print(f"⚠️ ONNX 모델 로드 실패: {e}")
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """얼굴 이미지를 모델 입력 형식으로 전처리"""
        # BGR -> Gray 변환
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        
        # 64x64로 리사이즈 (FERPlus 모델 입력 크기)
        face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        # 정규화 (0~1 범위)
        face_normalized = face_resized.astype('float32') / 255.0
        
        # 차원 추가: (1, 1, 64, 64) - NCHW 형식
        face_input = np.expand_dims(face_normalized, axis=0)  # 배치 차원
        face_input = np.expand_dims(face_input, axis=0)       # 채널 차원
        
        return face_input
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 함수"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """감정 예측
        
        Args:
            face_img: 얼굴 이미지 (BGR)
        
        Returns:
            (감정 레이블, 신뢰도)
        """
        # ONNX 모델이 로드되었으면 실제 추론
        if self.session is not None:
            try:
                # 전처리
                face_input = self.preprocess_face(face_img)
                
                # 추론
                outputs = self.session.run([self.output_name], 
                                          {self.input_name: face_input})
                
                # Softmax 적용
                probabilities = self.softmax(outputs[0][0])
                
                # 가장 높은 확률의 감정
                emotion_idx = np.argmax(probabilities)
                emotion = self.emotions[emotion_idx]
                confidence = float(probabilities[emotion_idx])
                
                return emotion, confidence
                
            except Exception as e:
                print(f"추론 중 오류: {e}")
                # 오류 시 더미 모드로 폴백
        
        # 더미 구현 (모델이 없거나 오류 시)
        return self._dummy_prediction(face_img)
    
    def _dummy_prediction(self, face_img: np.ndarray) -> Tuple[str, float]:
        """더미 예측 (모델 없을 때)"""
        # 얼굴 밝기 기반 간단한 예측
        brightness = np.mean(face_img)
        
        if brightness > 150:
            emotion = "Happy"
            confidence = 0.75
        elif brightness > 100:
            emotion = "Neutral" 
            confidence = 0.80
        elif brightness > 70:
            emotion = "Sad"
            confidence = 0.65
        else:
            emotion = "Anger"
            confidence = 0.60
            
        return emotion, confidence
    
    def get_emotion_probabilities(self, face_img: np.ndarray) -> Dict[str, float]:
        """모든 감정의 확률 반환"""
        if self.session is not None:
            try:
                face_input = self.preprocess_face(face_img)
                outputs = self.session.run([self.output_name], 
                                          {self.input_name: face_input})
                probabilities = self.softmax(outputs[0][0])
                
                return {emotion: float(prob) 
                       for emotion, prob in zip(self.emotions, probabilities)}
            except:
                pass
        
        # 더미 구현
        emotion, conf = self._dummy_prediction(face_img)
        probs = {e: 0.05 for e in self.emotions}
        probs[emotion] = conf
        return probs
    
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """감정별 색상 (BGR)"""
        colors = {
            'Happy': (0, 255, 0),      # 초록
            'Sad': (255, 100, 0),      # 파랑
            'Anger': (0, 0, 255),      # 빨강
            'Surprise': (0, 255, 255), # 노랑
            'Fear': (255, 0, 255),     # 보라
            'Disgust': (128, 0, 128),  # 자주
            'Contempt': (255, 128, 0), # 하늘
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
        
        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 배경 박스 (텍스트 위)
        padding = 5
        cv2.rectangle(frame, 
                     (x - padding, y - text_size[1] - padding * 2), 
                     (x + text_size[0] + padding, y), 
                     color, -1)
        
        # 텍스트 (흰색)
        cv2.putText(frame, text, 
                   (x, y - padding), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 얼굴 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # 신뢰도 바 (선택적)
        if confidence > 0:
            bar_height = int(h * confidence)
            bar_x = x + w + 5
            cv2.rectangle(frame, 
                         (bar_x, y + h - bar_height), 
                         (bar_x + 10, y + h), 
                         color, -1)
            cv2.rectangle(frame, 
                         (bar_x, y), 
                         (bar_x + 10, y + h), 
                         (200, 200, 200), 1)
        
        return frame
