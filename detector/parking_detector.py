import cv2
import numpy as np
import os
from config import PARKING_SPOTS, OCCUPANCY_THRESHOLD
from detector.color_utils import get_dominant_color, get_color_name

class ParkingDetector:
    """
    Detector de vagas que utiliza BackgroundSubtractorMOG2 para criar um modelo de fundo
    adaptativo. Se existir uma imagem de estacionamento vazio em 'assets/empty-parking.png',
    ela será usada para inicializar o modelo.
    """
    def __init__(self):
        self.spots = PARKING_SPOTS
        self.threshold = OCCUPANCY_THRESHOLD

        # 1) Cria o subtractor de fundo
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,        # quantos frames manter na memória
            varThreshold=50,    # sensibilidade à variação de cor
            detectShadows=False # desliga detecção de sombras (pixel=127)
        )

        # 2) Se existir imagem vazia, aplica várias vezes para estabilizar o modelo
        empty_path = os.path.join('assets', 'empty-parking.png')
        if os.path.exists(empty_path):
            empty_img = cv2.imread(empty_path)
            if empty_img is not None:
                for _ in range(30):
                    self.bg_subtractor.apply(empty_img)

    def detect(self, frame):
        results = []
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

        for (x, y, w, h) in self.spots:
            mask_roi = fg_mask[y:y+h, x:x+w]
            non_zero = cv2.countNonZero(mask_roi)
            occupied = non_zero > self.threshold

            color = None
            if occupied:
                spot_img = frame[y:y+h, x:x+w]
                # aqui passamos a máscara só com pixels do carro
                color = get_dominant_color(spot_img, mask_roi)

            results.append((occupied, color))
        return results

    def draw_annotations(self, frame, detections):
        total_spots = len(detections)
        free_spots = sum(1 for o, _ in detections if not o)

        for (x, y, w, h), (occupied, color) in zip(self.spots, detections):
            label = "Ocupada" if occupied else "Livre"
            rect_color = (0, 0, 255) if occupied else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y-th-4), (x+tw+4, y), rect_color, -1)
            cv2.putText(frame, label, (x+2, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if occupied and color is not None:
                # color é (R, G, B)
                bgr = (color[2], color[1], color[0])
                cv2.circle(frame, (x+10, y+10), 8, bgr, -1)

                hex_color = "#{:02X}{:02X}{:02X}".format(*color)
                name = get_color_name(color)  # ex: "white" ou "lightgray"

                # exibe hex na primeira linha e nome na segunda
                cv2.putText(frame, hex_color, (x+20, y+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, name, (x+20, y+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # totalizador (mesmo de antes) …
        total_text = f"Vagas livres: {free_spots}/{total_spots}"
        (tw, th), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10+tw+20, 10+th+20), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, total_text, (20, 10+th+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        return frame