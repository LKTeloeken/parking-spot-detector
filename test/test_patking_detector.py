import cv2
import numpy as np
import pytest
from detector.parking_detector import ParkingDetector

def test_empty_spot(tmp_path, monkeypatch):
    # cria um frame todo preto (vagas livres)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = ParkingDetector()
    results = detector.detect(frame, bg_frame=frame)
    assert all(not occupied for occupied, _ in results)

def test_occupied_spot(tmp_path):
    # frame com uma mancha branca simulando carro
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x, y, w, h = detector.spots[0]
    frame[y:y+h, x:x+w] = 255
    detector = ParkingDetector()
    results = detector.detect(frame, bg_frame=np.zeros_like(frame))
    assert results[0][0] is True