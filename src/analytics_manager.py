import mlflow
import mlflow.pytorch
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any
import json


class AnalyticsManager:
    """MLflow 실험 추적 및 로그 관리"""
    
    def __init__(self, experiment_name="video-quality-analytics"):
        # MLflow 설정
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        
        # 로그 디렉토리 설정
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # CSV 로거 초기화
        self.metrics_csv = os.path.join(self.log_dir, "metrics.csv")
        self.emotions_csv = os.path.join(self.log_dir, "emotions.csv")
        
        # 데이터 버퍼
        self.metrics_buffer = []
        self.emotions_buffer = []
        
        # MLflow 실행 시작
        self.run = None
        
    def start_run(self, run_name=None):
        """MLflow 실행 시작"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(run_name=run_name)
        
        # 실행 파라미터 기록
        mlflow.log_param("start_time", datetime.now().isoformat())
        mlflow.log_param("video_source", os.getenv("VIDEO_SOURCE", "0"))
        mlflow.log_param("quality_threshold", os.getenv("QUALITY_THRESHOLD", "0.6"))
        
        return self.run
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """메트릭 로깅"""
        # MLflow에 로깅
        if self.run:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        
        # CSV 버퍼에 추가
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_buffer.append(metrics)
        
        # 10개마다 CSV 저장
        if len(self.metrics_buffer) >= 10:
            self.save_metrics_to_csv()
    
    def log_emotion(self, emotion: str, confidence: float, face_id: int = 0):
        """감정 로깅"""
        emotion_data = {
            'timestamp': datetime.now().isoformat(),
            'face_id': face_id,
            'emotion': emotion,
            'confidence': confidence
        }
        
        self.emotions_buffer.append(emotion_data)
        
        # MLflow에 최근 감정 로깅
        if self.run:
            mlflow.log_metric(f"emotion_confidence_face_{face_id}", confidence)
    
    def save_metrics_to_csv(self):
        """메트릭 CSV 저장"""
        if self.metrics_buffer:
            df = pd.DataFrame(self.metrics_buffer)
            
            # 파일이 있으면 추가, 없으면 생성
            if os.path.exists(self.metrics_csv):
                df.to_csv(self.metrics_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(self.metrics_csv, index=False)
            
            self.metrics_buffer = []
    
    def save_emotions_to_csv(self):
        """감정 CSV 저장"""
        if self.emotions_buffer:
            df = pd.DataFrame(self.emotions_buffer)
            
            if os.path.exists(self.emotions_csv):
                df.to_csv(self.emotions_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(self.emotions_csv, index=False)
            
            self.emotions_buffer = []
    
    def log_quality_metrics(self, blur_score: float, brightness: float, 
                           noise_level: float, freeze_detected: bool):
        """품질 메트릭 통합 로깅"""
        quality_metrics = {
            'blur_score': blur_score,
            'brightness': brightness,
            'noise_level': noise_level,
            'freeze_detected': int(freeze_detected),
            'overall_quality': (1 - blur_score) * brightness / 255.0 * (1 - noise_level)
        }
        
        return quality_metrics
    
    def end_run(self):
        """MLflow 실행 종료"""
        # 남은 데이터 저장
        self.save_metrics_to_csv()
        self.save_emotions_to_csv()
        
        # MLflow 실행 종료
        if self.run:
            mlflow.end_run()
            self.run = None
    
    def __del__(self):
        """소멸자에서 자동 저장"""
        try:
            self.end_run()
        except:
            pass