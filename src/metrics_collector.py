import psutil
import time
import numpy as np
from typing import Dict, Any


class MetricsCollector:
    """시스템 및 비디오 메트릭 수집"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps_history = []
        
    def collect_system_metrics(self) -> Dict[str, float]:
        """CPU, 메모리 사용률 수집"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024)
        }
    
    def update_video_metrics(self) -> Dict[str, float]:
        """FPS, 지연시간 계산"""
        current_time = time.time()
        
        # FPS 계산
        time_diff = current_time - self.last_frame_time
        current_fps = 1.0 / time_diff if time_diff > 0 else 0
        
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 30:  # 최근 30프레임만 유지
            self.fps_history.pop(0)
        
        self.frame_count += 1
        self.last_frame_time = current_time
        
        # 평균 FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # 총 실행 시간
        elapsed_time = current_time - self.start_time
        
        return {
            'current_fps': current_fps,
            'avg_fps': avg_fps,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time,
            'latency_ms': time_diff * 1000
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """모든 메트릭 수집"""
        metrics = {}
        metrics.update(self.collect_system_metrics())
        metrics.update(self.update_video_metrics())
        metrics['timestamp'] = time.time()
        return metrics
    
    def reset(self):
        """메트릭 초기화"""
        self.start_time = time.time()
        self.frame_count = 0
        self.fps_history = []