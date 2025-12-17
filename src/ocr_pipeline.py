import cv2
import numpy as np
from typing import List, Dict, Any


class OCRPipeline:
    """OCR 텍스트 추출 (간단 버전)"""
    
    def __init__(self):
        # EasyOCR 대신 간단한 텍스트 영역 감지
        self.last_text = ""
        self.confidence = 0.0
        
    def extract_text(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임에서 텍스트 추출 (더미 구현)"""
        # 실제로는 EasyOCR 사용해야 하지만, 시연용으로 간단히
        
        # 텍스트 영역 감지 시뮬레이션
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출로 텍스트 영역 추정
        edges = cv2.Canny(gray, 50, 150)
        text_density = np.sum(edges > 0) / edges.size
        
        # 더미 텍스트 생성
        if text_density > 0.05:
            self.last_text = f"Text_Area_{int(text_density*1000)}"
            self.confidence = min(0.95, text_density * 10)
        else:
            self.last_text = ""
            self.confidence = 0.0
        
        return {
            'text': self.last_text,
            'confidence': self.confidence,
            'has_text': len(self.last_text) > 0,
            'text_density': text_density
        }
    
    def draw_text_regions(self, frame: np.ndarray) -> np.ndarray:
        """텍스트 영역 표시"""
        if self.confidence > 0:
            # 상단에 OCR 정보 표시
            text = f"OCR: {self.last_text} ({self.confidence:.2%})"
            cv2.putText(frame, text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame