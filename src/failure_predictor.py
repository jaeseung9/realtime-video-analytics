import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class FailurePredictor:
    """영상 품질 기반 이상/고장 예측"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.metrics_buffer = deque(maxlen=window_size)
        
        # 임계값 설정
        self.thresholds = {
            'quality_min': 0.4,
            'blur_max': 0.7,
            'brightness_min': 30,
            'brightness_max': 220,
            'noise_max': 0.6
        }
        
    def add_metrics(self, metrics: Dict):
        """메트릭 추가"""
        self.metrics_buffer.append(metrics)
    
    def calculate_features(self) -> Dict[str, float]:
        """슬라이딩 윈도우 특징 계산"""
        if len(self.metrics_buffer) < 5:
            return {}
        
        # 품질 점수 추출
        quality_scores = [m.get('quality_score', 0.5) for m in self.metrics_buffer]
        blur_scores = [m.get('blur_score', 0) for m in self.metrics_buffer]
        brightness = [m.get('brightness', 128) for m in self.metrics_buffer]
        
        features = {
            'quality_mean': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'quality_min': np.min(quality_scores),
            'blur_mean': np.mean(blur_scores),
            'brightness_mean': np.mean(brightness),
            'brightness_std': np.std(brightness),
        }
        
        # 추세 계산 (slope)
        if len(quality_scores) > 10:
            x = np.arange(len(quality_scores))
            quality_slope = np.polyfit(x, quality_scores, 1)[0]
            features['quality_trend'] = quality_slope
        else:
            features['quality_trend'] = 0
        
        return features
    
    def predict_failure(self) -> Tuple[float, str, str]:
        """고장 확률 예측
        Returns: (probability, status, reason)
        """
        if len(self.metrics_buffer) < 5:
            return 0.0, "Normal", "데이터 수집 중"
        
        features = self.calculate_features()
        
        # 규칙 기반 이상 점수 계산
        anomaly_score = 0.0
        reasons = []
        
        # 품질 점수 체크
        if features['quality_mean'] < self.thresholds['quality_min']:
            anomaly_score += 0.3
            reasons.append("낮은 품질")
        
        if features['quality_std'] > 0.2:
            anomaly_score += 0.2
            reasons.append("불안정한 품질")
        
        # 품질 하락 추세
        if features.get('quality_trend', 0) < -0.01:
            anomaly_score += 0.2
            reasons.append("품질 하락 중")
        
        # 블러 체크
        if features['blur_mean'] > self.thresholds['blur_max']:
            anomaly_score += 0.2
            reasons.append("심한 블러")
        
        # 밝기 체크
        if features['brightness_mean'] < self.thresholds['brightness_min']:
            anomaly_score += 0.1
            reasons.append("너무 어두움")
        elif features['brightness_mean'] > self.thresholds['brightness_max']:
            anomaly_score += 0.1
            reasons.append("과다 노출")
        
        # 확률로 변환
        failure_prob = min(1.0, anomaly_score)
        
        # 상태 결정
        if failure_prob >= 0.7:
            status = "Critical"
        elif failure_prob >= 0.5:
            status = "Warning"
        elif failure_prob >= 0.3:
            status = "Caution"
        else:
            status = "Normal"
        
        reason = ", ".join(reasons) if reasons else "정상"
        
        return failure_prob, status, reason
    
    def get_summary(self) -> Dict:
        """현재 상태 요약"""
        prob, status, reason = self.predict_failure()
        features = self.calculate_features()
        
        return {
            'failure_probability': prob,
            'status': status,
            'reason': reason,
            'features': features,
            'buffer_size': len(self.metrics_buffer)
        }