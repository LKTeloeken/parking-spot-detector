import cv2
from detector.parking_detector import ParkingDetector

def process_video(video_path, config_path=None):
    """
    Captura vídeo, processa frame a frame usando BackgroundSubtractor MOG2
    e exibe as anotações de vagas.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo: {video_path}")
        return

    detector = ParkingDetector()

    # Opcional: “aquece” o subtractor com alguns frames iniciais
    # for _ in range(30):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     detector.bg_subtractor.apply(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta ocupação e cor dominante
        detections = detector.detect(frame)

        # Desenha retângulos, labels e círculos de cor
        annotated = detector.draw_annotations(frame, detections)

        cv2.imshow("Parking Spot Detector", annotated)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()