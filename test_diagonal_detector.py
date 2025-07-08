import cv2
import numpy as np
from detector.polygon_parking_detector import PolygonParkingDetector
from config_diagonal import LAYOUT_CONFIG


def test_diagonal_detector():
    """
    Testa o detector com vagas na diagonal.
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

    # Ler primeiro frame para seleção
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o frame.")
        return

    print("=== TESTE DO DETECTOR DIAGONAL ===")
    print("Escolha o layout:")
    print("1. Layout diagonal pré-configurado")
    print("2. Layout paralelo na diagonal")
    print("3. Layout em V")
    print("4. Layout personalizado (V1,V2 diagonal ESQUERDA | V3,V4 diagonal DIREITA)")
    print("5. Seleção interativa")
    print("6. Testar todos os layouts")
    
    choice = input("Opção (1-6): ")
    
    if choice == "1":
        test_single_layout("diagonal", bg_frame, cap)
    elif choice == "2":
        test_single_layout("parallel", bg_frame, cap)
    elif choice == "3":
        test_single_layout("v_shape", bg_frame, cap)
    elif choice == "4":
        test_single_layout("custom", bg_frame, cap)
    elif choice == "5":
        test_interactive_selection(frame, bg_frame, cap)
    elif choice == "6":
        test_all_layouts(bg_frame, cap)
    else:
        print("Opção inválida. Usando layout personalizado.")
        test_single_layout("custom", bg_frame, cap)


def test_single_layout(layout_name, bg_frame, cap):
    """
    Testa um layout específico.
    """
    print(f"\n=== TESTANDO LAYOUT: {layout_name.upper()} ===")
    
    # Criar detector com layout específico
    polygons = LAYOUT_CONFIG[layout_name]
    detector = PolygonParkingDetector(polygons)
    
    # Reiniciar vídeo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Pressione 'q' para sair, 'p' para pausar, SPACE para próximo frame")
    
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
        info_text = f"Layout: {layout_name.upper()} | Frame: {frame_num}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Status das vagas
        status_text = "Status: "
        for i, (occupied, _) in enumerate(detections):
            status_text += f"V{i+1}:{'O' if occupied else 'L'} "
        cv2.putText(annotated, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        cv2.imshow(f"Detector Diagonal - {layout_name}", annotated)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Pausado' if paused else 'Retomado'}")
        elif key == ord(' '):  # Space
            if paused:
                paused = False
                continue
    
    cv2.destroyAllWindows()


def test_interactive_selection(frame, bg_frame, cap):
    """
    Testa seleção interativa de vagas.
    """
    print("\n=== SELEÇÃO INTERATIVA ===")
    
    # Criar detector vazio
    detector = PolygonParkingDetector([])
    
    # Usar seletor interativo
    polygons = detector.get_interactive_selector(frame)
    
    if len(polygons) == 0:
        print("Nenhum polígono selecionado.")
        return
    
    # Atualizar detector com polígonos selecionados
    detector = PolygonParkingDetector(polygons)
    
    # Testar com vídeo
    print(f"Testando com {len(polygons)} vagas selecionadas...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        detections = detector.detect(frame, bg_frame)
        annotated = detector.draw_annotations(frame, detections)
        
        cv2.putText(annotated, "Vagas Personalizadas", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Detector Personalizado", annotated)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def test_all_layouts(bg_frame, cap):
    """
    Testa todos os layouts lado a lado.
    """
    print("\n=== COMPARAÇÃO DE TODOS OS LAYOUTS ===")
    
    # Criar detectores para cada layout
    detectors = {}
    for layout_name, polygons in LAYOUT_CONFIG.items():
        detectors[layout_name] = PolygonParkingDetector(polygons)
    
    print("Pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Criar layout de comparação
        results = []
        
        for layout_name, detector in detectors.items():
            detections = detector.detect(frame, bg_frame)
            annotated = detector.draw_annotations(frame.copy(), detections)
            
            # Adicionar título
            cv2.putText(annotated, layout_name.upper(), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Redimensionar para comparação
            height = 300
            width = int(annotated.shape[1] * height / annotated.shape[0])
            resized = cv2.resize(annotated, (width, height))
            results.append(resized)
        
        # Combinar horizontalmente
        if len(results) >= 2:
            top_row = np.hstack(results[:2])
            if len(results) >= 3:
                bottom_row = np.hstack([results[2], np.zeros_like(results[2])])
                combined = np.vstack([top_row, bottom_row])
            else:
                combined = top_row
        else:
            combined = results[0]
        
        cv2.imshow("Comparação de Layouts", combined)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def visualize_layouts():
    """
    Visualiza os layouts disponíveis.
    """
    print("\n=== VISUALIZAÇÃO DOS LAYOUTS ===")
    
    # Criar imagem em branco para visualização
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for layout_name, polygons in LAYOUT_CONFIG.items():
        layout_img = img.copy()
        
        for i, polygon in enumerate(polygons):
            color = colors[i % len(colors)]
            cv2.polylines(layout_img, [polygon.astype(np.int32)], True, color, 2)
            cv2.fillPoly(layout_img, [polygon.astype(np.int32)], color, lineType=cv2.LINE_AA)
            
            # Adicionar número da vaga
            center_x = int(np.mean(polygon[:, 0]))
            center_y = int(np.mean(polygon[:, 1]))
            cv2.putText(layout_img, f"V{i+1}", (center_x-10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Adicionar título
        cv2.putText(layout_img, f"Layout: {layout_name.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(f"Layout {layout_name}", layout_img)
    
    print("Pressione qualquer tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== DETECTOR DE VAGAS NA DIAGONAL ===")
    print("1. Visualizar layouts disponíveis")
    print("2. Testar detector")
    
    choice = input("Escolha (1-2): ")
    
    if choice == "1":
        visualize_layouts()
    elif choice == "2":
        test_diagonal_detector()
    else:
        print("Opção inválida. Mostrando layouts...")
        visualize_layouts() 