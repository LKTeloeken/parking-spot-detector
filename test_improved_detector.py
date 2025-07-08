import cv2
import numpy as np
from detector.improved_parking_detector import ImprovedParkingDetector


def test_improved_detector():
    """
    Testa o detector melhorado com calibração automática.
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

    # Coletar frames de amostra para calibração
    print("Coletando frames de amostra para calibração...")
    sample_frames = []
    frame_count = 0
    
    while len(sample_frames) < 10:  # Coletar 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Coletar um frame a cada 30 frames
        if frame_count % 30 == 0:
            sample_frames.append(frame.copy())
        
        frame_count += 1
    
    if len(sample_frames) < 5:
        print("Erro: Não foi possível coletar frames suficientes para calibração.")
        return
    
    # Criar detector melhorado
    detector = ImprovedParkingDetector()
    
    # Calibrar thresholds
    print("Calibrando thresholds...")
    detector.calibrate_thresholds(bg_frame, sample_frames)
    
    # Reiniciar vídeo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\nIniciando detecção melhorada...")
    print("Pressione 'q' para sair, 'p' para pausar, 'r' para recalibrar")
    
    paused = False
    frame_num = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Reiniciar vídeo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_num += 1
        
        # Detecção
        detections = detector.detect(frame, bg_frame)
        annotated = detector.draw_annotations(frame, detections)
        
        # Adicionar informações no frame
        info_text = f"Frame: {frame_num} | Detector: Melhorado"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Mostrar status das vagas
        status_text = "Status: "
        for i, (occupied, _) in enumerate(detections):
            status_text += f"V{i+1}:{'O' if occupied else 'L'} "
        cv2.putText(annotated, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Mostrar instruções
        cv2.putText(annotated, "q:sair p:pausar r:recalibrar", (10, annotated.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Improved Parking Detector", annotated)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Pausado' if paused else 'Retomado'}")
        elif key == ord('r'):
            print("Recalibrando...")
            # Coletar novos frames para recalibração
            sample_frames = []
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            for i in range(10):
                ret, sample_frame = cap.read()
                if ret:
                    sample_frames.append(sample_frame.copy())
            
            if len(sample_frames) > 0:
                detector.calibrate_thresholds(bg_frame, sample_frames)
            
            # Voltar à posição original
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()


def compare_detectors():
    """
    Compara o detector original com o melhorado.
    """
    from detector.parking_detector import ParkingDetector
    
    # Carregar imagens
    bg_frame = cv2.imread("assets/EstacionamentoVazio.png")
    if bg_frame is None:
        print("Erro: Não foi possível carregar o frame de fundo.")
        return

    cap = cv2.VideoCapture("assets/Estacionamento.mp4")
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo.")
        return

    # Ler primeiro frame
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o frame.")
        return
    
    # Criar detectores
    original_detector = ParkingDetector()
    improved_detector = ImprovedParkingDetector()
    
    # Calibrar detector melhorado
    sample_frames = [frame]
    for i in range(9):
        ret, sample_frame = cap.read()
        if ret:
            sample_frames.append(sample_frame)
    
    improved_detector.calibrate_thresholds(bg_frame, sample_frames)
    
    # Comparar detecções
    original_detections = original_detector.detect(frame, bg_frame)
    improved_detections = improved_detector.detect(frame, bg_frame)
    
    # Anotar frames
    original_annotated = original_detector.draw_annotations(frame.copy(), original_detections)
    improved_annotated = improved_detector.draw_annotations(frame.copy(), improved_detections)
    
    # Adicionar títulos
    cv2.putText(original_annotated, "DETECTOR ORIGINAL", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(improved_annotated, "DETECTOR MELHORADO", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar resultados
    print("=== COMPARAÇÃO DOS DETECTORES ===")
    print("Detector Original:")
    for i, (occupied, _) in enumerate(original_detections):
        print(f"  Vaga {i+1}: {'OCUPADA' if occupied else 'LIVRE'}")
    
    print("\nDetector Melhorado:")
    for i, (occupied, _) in enumerate(improved_detections):
        print(f"  Vaga {i+1}: {'OCUPADA' if occupied else 'LIVRE'}")
    
    # Exibir imagens
    cv2.imshow("Original", original_annotated)
    cv2.imshow("Melhorado", improved_annotated)
    
    print("\nPressione qualquer tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()


if __name__ == "__main__":
    print("=== TESTE DO DETECTOR MELHORADO ===")
    print("1. Teste do detector melhorado")
    print("2. Comparação entre detectores")
    
    choice = input("Escolha uma opção (1 ou 2): ")
    
    if choice == "1":
        test_improved_detector()
    elif choice == "2":
        compare_detectors()
    else:
        print("Opção inválida. Executando teste padrão...")
        test_improved_detector() 