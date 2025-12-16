import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """OpenCV Haar Cascade 기반 얼굴 검출기"""
    
    def __init__(self):
        """얼굴 검출기 초기화"""
        import os
        import tempfile
        
        # 여러 경로를 시도 (한글 경로 문제 고려)
        cascade_paths = [
            # 방법 1: 프로젝트 resources 폴더 (상대 경로)
            os.path.join('resources', 'haarcascade_frontalface_default.xml'),
            # 방법 2: src 기준 상대 경로
            os.path.join('..', 'resources', 'haarcascade_frontalface_default.xml'),
            # 방법 3: 현재 스크립트 기준 절대 경로
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'haarcascade_frontalface_default.xml'),
        ]
        
        self.face_cascade = None
        self.temp_file = None
        
        for path in cascade_paths:
            try:
                # 경로 정규화
                normalized_path = os.path.normpath(os.path.abspath(path))
                print(f"경로 시도: {normalized_path}")
                print(f"파일 존재: {os.path.exists(normalized_path)}")
                
                if os.path.exists(normalized_path):
                    # 한글 경로 우회: 임시 파일 사용
                    # XML 파일을 바이트로 읽기
                    with open(normalized_path, 'rb') as f:
                        file_data = f.read()
                    
                    print(f"파일 크기: {len(file_data)} bytes")
                    
                    # 임시 파일명으로 저장 (영문 경로)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='wb')
                    tmp.write(file_data)
                    tmp.close()
                    tmp_path = tmp.name
                    
                    print(f"임시 파일: {tmp_path}")
                    
                    # Cascade 로드
                    cascade = cv2.CascadeClassifier(tmp_path)
                    
                    print(f"Cascade empty: {cascade.empty()}")
                    
                    if not cascade.empty():
                        self.face_cascade = cascade
                        self.temp_file = tmp_path  # 나중에 삭제하기 위해 저장
                        print(f"✅ 얼굴 검출기 로드 완료: {normalized_path}")
                        return  # 성공하면 바로 종료
                    else:
                        # 실패하면 임시 파일 삭제
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                        
            except Exception as e:
                print(f"❌ 경로 시도 실패 ({path}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 모든 경로 실패
        raise Exception(
            "얼굴 검출 모델을 로드할 수 없습니다!\n"
            "resources/haarcascade_frontalface_default.xml 파일을 다운로드해주세요.\n"
            "다운로드: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        )
    
    def detect_faces(self, frame: np.ndarray, min_neighbors: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        프레임에서 얼굴 검출
        
        Args:
            frame: 입력 프레임 (BGR)
            min_neighbors: 최소 이웃 개수 (낮을수록 민감)
        
        Returns:
            얼굴 바운딩 박스 리스트 [(x, y, w, h), ...]
        """
        if frame is None or frame.size == 0:
            return []
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 히스토그램 평활화 (조명 변화에 강인하게)
        gray = cv2.equalizeHist(gray)
        
        # 얼굴 검출 - 파라미터 완화
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,     # 1.1 → 1.05 (더 세밀하게 스케일링)
            minNeighbors=min_neighbors,  # 사용자 지정 가능
            minSize=(20, 20),     # 30 → 20 (더 작은 얼굴도 검출)
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 중복 제거 (겹치는 박스 통합)
        if len(faces) > 1:
            faces = self._merge_overlapping_boxes(faces)
        
        # NumPy array를 리스트로 변환
        return [tuple(face) for face in faces]
    
    def _merge_overlapping_boxes(self, boxes: np.ndarray, overlap_threshold: float = 0.3) -> np.ndarray:
        """겹치는 바운딩 박스 병합"""
        if len(boxes) == 0:
            return boxes
        
        # 면적 기준 정렬
        boxes = sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)
        merged = []
        
        while len(boxes) > 0:
            current = boxes[0]
            boxes = boxes[1:]
            
            # 현재 박스와 겹치지 않는 박스만 남김
            non_overlapping = []
            for box in boxes:
                if self._calculate_iou(current, box) < overlap_threshold:
                    non_overlapping.append(box)
            
            merged.append(current)
            boxes = non_overlapping
        
        return np.array(merged)
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """두 박스의 IoU(Intersection over Union) 계산"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 교집합 영역
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 합집합 영역
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # IoU
        return inter_area / union_area if union_area > 0 else 0
    
    def draw_faces(
        self, 
        frame: np.ndarray, 
        faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        프레임에 얼굴 바운딩 박스 그리기
        
        Args:
            frame: 입력 프레임
            faces: 얼굴 바운딩 박스 리스트
            color: 박스 색상 (BGR)
            thickness: 선 두께
        
        Returns:
            바운딩 박스가 그려진 프레임
        """
        frame_copy = frame.copy()
        
        for (x, y, w, h) in faces:
            # 사각형 그리기
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
            
            # 얼굴 번호 표시
            cv2.putText(
                frame_copy,
                f"Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return frame_copy
    
    def get_face_count(self, faces: List) -> int:
        """검출된 얼굴 개수 반환"""
        return len(faces)