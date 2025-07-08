import cv2
from detector.polygon_parking_detector import PolygonParkingDetector
from config_diagonal import PARKING_SPOTS_CUSTOM


def main():
    """
    Função principal para detectar vagas de estacionamento usando layout personalizado.
    """
    # Carregar imagens
    bg_frame = cv2.imread("assets/EstacionamentoVazio.png")
    if bg_frame is None:
        print("Erro: Não foi possível carregar o frame de fundo.")
        return

    cap = cv2.VideoCapture("assets/Estacionamento.mp4")
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo.")
        return

    # Criar detector com layout personalizado
    detector = PolygonParkingDetector(PARKING_SPOTS_CUSTOM)
    
    print("=== DETECTOR DE VAGAS - LAYOUT PERSONALIZADO ===")
    print("V1: Formato trapézio (topo maior que base)")
    print("V2: Largura reduzida (mais estreito)")
    print("V3: Diagonal para direita")
    print("V4: Diagonal para direita")
    print("\nControles:")
    print("- 'q' ou ESC: Sair")
    print("- 'p': Pausar/Despausar")
    print("- 'r': Reiniciar vídeo")
    print("-" * 50)
    
    paused = False
    frame_num = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Reiniciar vídeo automaticamente
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                continue
            frame_num += 1
        
        # Detecção
        detections = detector.detect(frame, bg_frame)
        annotated = detector.draw_annotations(frame, detections)
        
        # Informações na tela
        info_text = f"Frame: {frame_num} | Layout: Personalizado"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Status das vagas
        status_text = "Status: "
        occupied_count = 0
        for i, (occupied, _) in enumerate(detections):
            status = 'O' if occupied else 'L'
            status_text += f"V{i+1}:{status} "
            if occupied:
                occupied_count += 1
        
        cv2.putText(annotated, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Contador de vagas
        summary_text = f"Ocupadas: {occupied_count}/{len(detections)} | Livres: {len(detections) - occupied_count}"
        cv2.putText(annotated, summary_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        
        # Instruções
        cv2.putText(annotated, "q:sair p:pausar r:reiniciar", (10, annotated.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Parking Spot Detector - Layout Personalizado", annotated)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' ou ESC
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSADO" if paused else "EXECUTANDO"
            print(f"Status: {status}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            print("Vídeo reiniciado")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detector finalizado.")


if __name__ == "__main__":
    main()
