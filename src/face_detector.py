import cv2
import numpy as np
from typing import List, Tuple


class FaceDetector:
    """OpenCV Haar Cascade 기반 얼굴 검출기 (오검출 억제 버전)"""

    def __init__(self):
        import os
        import shutil
        import time

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        face_xml = os.path.join(project_root, "resources", "haarcascade_frontalface_default.xml")
        eye_xml = os.path.join(project_root, "resources", "haarcascade_eye_tree_eyeglasses.xml")

        if not os.path.exists(face_xml):
            raise FileNotFoundError(f"얼굴 XML 없음: {face_xml}")
        if not os.path.exists(eye_xml):
            raise FileNotFoundError(f"눈 XML 없음: {eye_xml}")

        # 한글 경로 우회: C:\opencv_temp 에 복사해서 로드
        drive = os.path.splitdrive(os.path.abspath(__file__))[0]  # C:
        temp_dir = os.path.join(drive, os.sep, "opencv_temp")
        os.makedirs(temp_dir, exist_ok=True)

        ts = int(time.time() * 1000)
        self.temp_face = os.path.join(temp_dir, f"face_{ts}.xml")
        self.temp_eye = os.path.join(temp_dir, f"eye_{ts}.xml")

        shutil.copy2(face_xml, self.temp_face)
        shutil.copy2(eye_xml, self.temp_eye)

        self.face_cascade = cv2.CascadeClassifier(self.temp_face)
        self.eye_cascade = cv2.CascadeClassifier(self.temp_eye)

        if self.face_cascade.empty():
            raise RuntimeError("❌ 얼굴 cascade 로드 실패 (XML 파싱 실패)")
        if self.eye_cascade.empty():
            raise RuntimeError("❌ 눈 cascade 로드 실패 (XML 파싱 실패)")

        print("✅ FaceDetector 로드 성공")

    def __del__(self):
        import os
        for p in [getattr(self, "temp_face", None), getattr(self, "temp_eye", None)]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass

    def detect_faces(
        self,
        frame: np.ndarray,
        min_neighbors: int = 7,        # 기본값을 더 엄격하게
        scale_factor: float = 1.2,     # 기본값을 더 안정적으로
        min_size: Tuple[int, int] = None,
        weight_threshold: float = 2.5, # 너무 낮으면 오검출 증가
        require_eye: bool = True       # 오검출 방지 핵심
    ) -> List[Tuple[int, int, int, int]]:

        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # 너무 작은 얼굴은 오검출이 많아서, 프레임 크기 기반으로 최소 크기 자동 설정
        if min_size is None:
            ms = max(60, int(min(w, h) * 0.08))  # 대략 화면의 8% 이상
            min_size = (ms, ms)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 어두운 프레임에서만 히스토그램 보정 (밝은 곳에서 오검출 방지)
        if gray.mean() < 80:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # ✅ detectMultiScale3: rejectLevels/levelWeights를 이용해 오검출 억제
        # (환경에 따라 detectMultiScale3가 없을 수도 있어서 fallback 포함)
        faces = []
        try:
            rects, rejectLevels, levelWeights = self.face_cascade.detectMultiScale3(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True
            )

            # weight로 1차 필터링
            for (x, y, ww, hh), wt in zip(rects, levelWeights):
                if wt >= weight_threshold:
                    faces.append((int(x), int(y), int(ww), int(hh)))

        except Exception:
            rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = [(int(x), int(y), int(ww), int(hh)) for (x, y, ww, hh) in rects]

        # ✅ 눈 검출로 2차 검증 (빈 공간 오검출 제거에 매우 효과적)
        if require_eye and len(faces) > 0:
            verified = []
            for (x, y, ww, hh) in faces:
                roi = gray[y:y + hh, x:x + ww]
                if roi.size == 0:
                    continue

                eyes = self.eye_cascade.detectMultiScale(
                    roi,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(15, 15)
                )

                if len(eyes) >= 1:  # 1개라도 있으면 얼굴로 인정
                    verified.append((x, y, ww, hh))

            faces = verified

        return faces

    def draw_faces(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        out = frame.copy()
        for i, (x, y, w, h) in enumerate(faces, 1):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, f"Face {i}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out
