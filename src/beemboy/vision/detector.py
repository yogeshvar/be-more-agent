from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectedFace:
    left: int
    top: int
    width: int
    height: int
    crop_bytes: bytes
    confidence: float = 1.0


class FaceDetector:
    """Best-effort face detector with optional OpenCV backend.

    The project keeps OpenCV optional so Pi deployments can choose their own
    detector stack. If OpenCV is unavailable, detection returns no faces.
    """

    def __init__(self, *, backend: str = "opencv", min_face_size_px: int = 80) -> None:
        self._backend = backend
        self._min_face_size_px = max(24, int(min_face_size_px))

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        if self._backend != "opencv":
            return []
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except Exception:
            return []
        if not image_bytes:
            return []
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self._min_face_size_px, self._min_face_size_px),
        )
        out: list[DetectedFace] = []
        for x, y, w, h in faces:
            crop = frame[y : y + h, x : x + w]
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            out.append(DetectedFace(left=int(x), top=int(y), width=int(w), height=int(h), crop_bytes=bytes(encoded)))
        return out
