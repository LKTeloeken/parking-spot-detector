import cv2
from detector.parking_detector import ParkingDetector


def process_video() -> None:
    """
    Processa o vídeo e detecta vagas de estacionamento.
    """
    bg_frame = cv2.imread("assets/EstacionamentoVazio.png")
    if bg_frame is None:
        print("Erro: Não foi possível carregar o frame de fundo.")
        return

    cap = cv2.VideoCapture("assets/Estacionamento.mp4")
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo.")
        return

    # # Redimensiona o frame de fundo para o tamanho do vídeo
    # bg_frame = cv2.resize(bg_frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    #     cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    detector = ParkingDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, bg_frame=bg_frame)
        annotated = detector.draw_annotations(frame, detections)

        cv2.imshow("Parking Spot Detector", annotated)
        key = cv2.waitKey(30) & 0xFF
        # ESC para sair
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()