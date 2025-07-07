import cv2
import numpy as np
from config import PARKING_SPOTS, OCCUPANCY_THRESHOLD
from detector.color_utils import get_dominant_color


class ParkingDetector:
    def __init__(self):
        self.spots = PARKING_SPOTS

    def detect(self, frame: np.ndarray, bg_frame: np.ndarray=None) -> list:
        """
        Retorna lista de tuplas (status, cor) por vaga:
          - status: True se ocupada, False caso contrário
          - cor: tupla RGB/HSV da cor dominante quando ocupada
        """
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for idx, (x, y, w, h) in enumerate(self.spots):
            roi = gray[y:y+h, x:x+w]
            occupied = False

            if bg_frame is not None:
                bg_roi = cv2.cvtColor(bg_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(bg_roi, roi)
                non_zero = cv2.countNonZero(diff)
                occupied = non_zero >= OCCUPANCY_THRESHOLD
            else:
                _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
                non_zero = cv2.countNonZero(thresh)
                occupied = non_zero >= OCCUPANCY_THRESHOLD

            color = None
            if occupied:
                spot_img = frame[y:y+h, x:x+w]
                color = get_dominant_color(spot_img)
            results.append((occupied, color))

        return results

    def draw_annotations(self, frame: np.ndarray, detections: tuple) -> np.ndarray:
        """
        Desenha retângulos e cores no frame.

        Parâmetros:
            frame: Frame do vídeo onde as anotações serão desenhadas.
            detections: Lista de tuplas (status, cor) para cada vaga.
                        - status: True se ocupada, False caso contrário
                        - cor: Tupla RGB/HSV da cor dominante quando ocupada
        
        Retorna:
            frame: Frame com as anotações desenhadas.
        """
        for idx, ((x, y, w, h), (occupied, color)) in enumerate(zip(self.spots, detections)):
            label = "Ocupada" if occupied else "Livre"
            color_box = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_box, 2)
            cv2.putText(frame, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            
            if color:
                cv2.circle(frame, (x + 10, y + 10), 8, color, -1)

        return frame
