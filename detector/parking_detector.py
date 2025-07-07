import cv2
import numpy as np
from config import PARKING_SPOTS, OCCUPANCY_THRESHOLD
from detector.color_utils import get_dominant_color

class ParkingDetector:
    def __init__(self):
        self.spots = PARKING_SPOTS

    def detect(self, frame, bg_frame=None):
        """
        Retorna lista de tuplas (status, cor) por vaga:
          - status: True se ocupada, False caso contrário
          - cor: tupla RGB/HSV da cor dominante quando ocupada
        """
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in self.spots:
            roi = gray[y:y+h, x:x+w]
            # comparativo com background ou limiar global
            occupied = False
            if bg_frame is not None:
                bg_roi = cv2.cvtColor(bg_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(bg_roi, roi)
                non_zero = cv2.countNonZero(diff)
                occupied = non_zero > OCCUPANCY_THRESHOLD
            else:
                _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
                occupied = cv2.countNonZero(thresh) > OCCUPANCY_THRESHOLD

            color = None
            if occupied:
                spot_img = frame[y:y+h, x:x+w]
                color = get_dominant_color(spot_img)
            results.append((occupied, color))
        return results

    def draw_annotations(self, frame, detections):
        """
        Desenha retângulos e cores no frame.
        """
        for idx, ((x, y, w, h), (occupied, color)) in enumerate(zip(self.spots, detections)):
            label = "Ocupada" if occupied else "Livre"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0) if not occupied else (0,0,255), 2)
            cv2.putText(frame, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if color:
                # mostre um pequeno círculo com a cor dominante
                cv2.circle(frame, (x + 10, y + 10), 8, color, -1)
        return frame