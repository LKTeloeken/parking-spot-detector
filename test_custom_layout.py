import cv2
import numpy as np
from detector.polygon_parking_detector import PolygonParkingDetector
from config_diagonal import PARKING_SPOTS_CUSTOM


def test_custom_layout():
    """
    Testa o layout personalizado com V1,V2 diagonal direita e V3,V4 diagonal esquerda.
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
    
    print("=== LAYOUT PERSONALIZADO ===")
    print("V1: FORMATO TRAPÉZIO (topo maior que base)")
    print("V2: LARGURA REDUZIDA (mais estreito)") 
    print("V3: Diagonal levemente para DIREITA (largura 200)")
    print("V4: Diagonal levemente para DIREITA (largura 150)")
    print("\nPressione 'q' para sair, 'p' para pausar")
    
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
        
        # Adicionar informações
        info_text = f"Layout Personalizado | Frame: {frame_num}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Status das vagas
        status_text = "Status: "
        for i, (occupied, _) in enumerate(detections):
            status_text += f"V{i+1}:{'O' if occupied else 'L'} "
        cv2.putText(annotated, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Instruções
        cv2.putText(annotated, "q:sair p:pausar", (10, annotated.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Detector Layout Personalizado", annotated)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Pausado' if paused else 'Retomado'}")
    
    cap.release()
    cv2.destroyAllWindows()


def visualize_custom_layout():
    """
    Visualiza apenas o layout personalizado.
    """
    print("=== VISUALIZAÇÃO DO LAYOUT PERSONALIZADO ===")
    
    # Criar imagem em branco
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, polygon in enumerate(PARKING_SPOTS_CUSTOM):
        color = colors[i % len(colors)]
        
        # Desenhar polígono
        cv2.polylines(img, [polygon.astype(np.int32)], True, color, 3)
        
        # Preencher com transparência
        overlay = img.copy()
        cv2.fillPoly(overlay, [polygon.astype(np.int32)], color)
        cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
        
        # Adicionar número da vaga
        center_x = int(np.mean(polygon[:, 0]))
        center_y = int(np.mean(polygon[:, 1]))
        cv2.putText(img, f"V{i+1}", (center_x-15, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Indicar formato da vaga
        if i == 0:  # V1
            direction = "▽"  # Trapézio (topo maior)
        elif i == 1:  # V2
            direction = "||"  # Estreito
        else:      # V3 e V4
            direction = "→"  # Diagonal direita
        
        cv2.putText(img, direction, (center_x-10, center_y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Adicionar título e legendas
    cv2.putText(img, "Layout Personalizado", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(img, "V1: TRAPÉZIO (topo > base)", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(img, "V2: LARGURA REDUZIDA | V3,V4: DIAGONAL DIREITA", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Layout Personalizado", img)
    
    print("Pressione qualquer tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== TESTE DO LAYOUT PERSONALIZADO ===")
    print("1. Visualizar layout")
    print("2. Testar com vídeo")
    
    choice = input("Escolha (1-2): ")
    
    if choice == "1":
        visualize_custom_layout()
    elif choice == "2":
        test_custom_layout()
    else:
        print("Opção inválida. Mostrando visualização...")
        visualize_custom_layout() 