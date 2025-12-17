import cv2
import numpy as np
from typing import Dict, Any


class QualityAnalyzer:
    """영상 품질 분석기"""
    
    def __init__(self):
        self.prev_frame = None
        self.freeze_counter = 0
        self.freeze_threshold = 5  # 5프레임 이상 동일하면 프리즈
        
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 품질 분석"""
        # 프레임 복사본으로 작업
        frame_copy = frame.copy()
        
        # 그레이스케일 변환
        if len(frame_copy.shape) == 3:
            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_copy
        
        # 1. 블러 감지 (Laplacian variance)
        blur_score = self.detect_blur(gray)
        
        # 2. 밝기 측정
        brightness = self.measure_brightness(gray)
        
        # 3. 노이즈 레벨
        noise_level = self.estimate_noise(gray)
        
        # 4. 프리즈 감지
        is_frozen = self.detect_freeze(gray)
        
        # 5. 전체 품질 점수 (0-1)
        quality_score = self.calculate_quality_score(
            blur_score, brightness, noise_level, is_frozen
        )
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'noise_level': noise_level,
            'is_frozen': is_frozen,
            'quality_score': quality_score,
            'quality_status': self.get_quality_status(quality_score)
        }
    
    def detect_blur(self, gray_frame: np.ndarray) -> float:
        """블러 감지 (낮을수록 선명)"""
        try:
            laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
            variance = laplacian.var()
            
            # 정규화 (0-1, 1이 가장 블러)
            blur_normalized = max(0, min(1, 1 - (variance / 1000)))
            return blur_normalized
        except:
            return 0.5
    
    def measure_brightness(self, gray_frame: np.ndarray) -> float:
        """밝기 측정 (0-255)"""
        try:
            return np.mean(gray_frame)
        except:
            return 128.0
    
    def estimate_noise(self, gray_frame: np.ndarray) -> float:
        """노이즈 추정 (0-1, 높을수록 노이즈 많음)"""
        try:
            # 고주파 성분으로 노이즈 추정
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
            
            filtered = cv2.filter2D(gray_frame, -1, kernel)
            noise = np.std(filtered)
            
            # 정규화
            noise_normalized = min(1.0, noise / 50.0)
            return noise_normalized
        except:
            return 0.0
    
    def detect_freeze(self, gray_frame: np.ndarray) -> bool:
        """프리즈 감지"""
        try:
            if self.prev_frame is None:
                self.prev_frame = gray_frame.copy()
                return False
            
            # 크기가 다르면 현재 프레임 크기에 맞춰 리사이즈
            if self.prev_frame.shape != gray_frame.shape:
                # 이전 프레임을 현재 프레임 크기로 리사이즈
                self.prev_frame = cv2.resize(self.prev_frame, 
                                            (gray_frame.shape[1], gray_frame.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
            
            # 타입 일치
            if self.prev_frame.dtype != gray_frame.dtype:
                self.prev_frame = self.prev_frame.astype(gray_frame.dtype)
            
            # 프레임 차이 계산
            diff = cv2.absdiff(self.prev_frame, gray_frame)
            mean_diff = np.mean(diff)
            
            # 차이가 거의 없으면 프리즈 카운터 증가
            if mean_diff < 2.0:
                self.freeze_counter += 1
            else:
                self.freeze_counter = 0
            
            # 현재 프레임을 이전 프레임으로 저장
            self.prev_frame = gray_frame.copy()
            
            return self.freeze_counter >= self.freeze_threshold
            
        except Exception as e:
            print(f"Freeze detection error: {e}")
            self.prev_frame = gray_frame.copy()
            return False
    
    def calculate_quality_score(self, blur: float, brightness: float, 
                               noise: float, frozen: bool) -> float:
        """전체 품질 점수 계산 (0-1, 높을수록 좋음)"""
        try:
            # 블러 점수 (선명할수록 좋음)
            blur_quality = 1 - blur
            
            # 밝기 점수 (50-200 범위가 좋음)
            if 50 <= brightness <= 200:
                brightness_quality = 1.0
            elif brightness < 50:
                brightness_quality = brightness / 50.0
            else:
                brightness_quality = max(0, 1 - (brightness - 200) / 55.0)
            
            # 노이즈 점수 (적을수록 좋음)
            noise_quality = 1 - noise
            
            # 프리즈 페널티
            freeze_penalty = 0.5 if frozen else 1.0
            
            # 가중 평균
            quality = (blur_quality * 0.3 + 
                      brightness_quality * 0.3 + 
                      noise_quality * 0.2 + 
                      freeze_penalty * 0.2)
            
            return max(0, min(1, quality))
        except:
            return 0.5
    
    def get_quality_status(self, score: float) -> str:
        """품질 상태 텍스트"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"