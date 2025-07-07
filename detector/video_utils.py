import cv2
from detector.parking_detector import ParkingDetector

def process_video(video_path, config_path):
    # carrega background do primeiro frame
    cap = cv2.VideoCapture(video_path)
    ret, bg_frame = cap.read()
    if not ret:
        print("Não foi possível ler o vídeo.")
        return

    detector = ParkingDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, bg_frame=bg_frame)
        annotated = detector.draw_annotations(frame, detections)

        cv2.imshow("Parking Spot Detector", annotated)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()