import cv2
import numpy as np
from config import PARKING_SPOTS, OCCUPANCY_THRESHOLD
from detector.parking_detector import ParkingDetector
from detector.color_utils import get_dominant_color


def analyze_parking_detection():
    """
    Analisa a detecção de vagas e mostra informações de debug.
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

    # Ler primeiro frame
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o frame.")
        return

    # Converter para escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)

    print("=== ANÁLISE DE DETECÇÃO DE VAGAS ===")
    print(f"Threshold atual: {OCCUPANCY_THRESHOLD}")
    print(f"Número de vagas: {len(PARKING_SPOTS)}")
    print("-" * 60)

    for idx, (x, y, w, h) in enumerate(PARKING_SPOTS):
        print(f"\nVAGA {idx + 1}: Posição ({x}, {y}), Tamanho ({w}x{h})")
        
        # Extrair ROIs
        roi_frame = frame_gray[y:y+h, x:x+w]
        roi_bg = bg_gray[y:y+h, x:x+w]
        
        # Calcular diferença absoluta
        diff = cv2.absdiff(roi_bg, roi_frame)
        non_zero_pixels = cv2.countNonZero(diff)
        
        # Calcular médias
        mean_frame = np.mean(roi_frame)
        mean_bg = np.mean(roi_bg)
        mean_diff = np.mean(diff)
        
        # Calcular desvio padrão
        std_frame = np.std(roi_frame)
        std_bg = np.std(roi_bg)
        
        # Detectar ocupação
        occupied = non_zero_pixels >= OCCUPANCY_THRESHOLD
        
        print(f"  Pixels diferentes: {non_zero_pixels}")
        print(f"  Média frame atual: {mean_frame:.2f}")
        print(f"  Média background: {mean_bg:.2f}")
        print(f"  Diferença média: {mean_diff:.2f}")
        print(f"  Desvio padrão frame: {std_frame:.2f}")
        print(f"  Desvio padrão background: {std_bg:.2f}")
        print(f"  Status: {'OCUPADA' if occupied else 'LIVRE'}")
        
        # Análise adicional - diferença percentual
        if mean_bg > 0:
            percent_diff = abs(mean_frame - mean_bg) / mean_bg * 100
            print(f"  Diferença percentual: {percent_diff:.2f}%")
        
        # Teste com diferentes thresholds
        print(f"  Testes de threshold:")
        for threshold in [100, 200, 300, 500, 1000, 2000]:
            test_occupied = non_zero_pixels >= threshold
            print(f"    Threshold {threshold}: {'OCUPADA' if test_occupied else 'LIVRE'}")

    cap.release()
    
    # Mostrar imagens para análise visual
    display_analysis(frame, bg_frame, frame_gray, bg_gray)


def display_analysis(frame, bg_frame, frame_gray, bg_gray):
    """
    Mostra as imagens para análise visual.
    """
    # Criar imagem de diferença
    diff_img = cv2.absdiff(bg_gray, frame_gray)
    
    # Desenhar ROIs nas imagens
    frame_with_rois = frame.copy()
    bg_with_rois = bg_frame.copy()
    diff_with_rois = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)
    
    for idx, (x, y, w, h) in enumerate(PARKING_SPOTS):
        # Desenhar retângulos
        cv2.rectangle(frame_with_rois, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(bg_with_rois, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(diff_with_rois, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Adicionar números das vagas
        cv2.putText(frame_with_rois, str(idx+1), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bg_with_rois, str(idx+1), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(diff_with_rois, str(idx+1), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Redimensionar para exibir
    height = 400
    frame_resized = cv2.resize(frame_with_rois, (int(frame_with_rois.shape[1] * height / frame_with_rois.shape[0]), height))
    bg_resized = cv2.resize(bg_with_rois, (int(bg_with_rois.shape[1] * height / bg_with_rois.shape[0]), height))
    diff_resized = cv2.resize(diff_with_rois, (int(diff_with_rois.shape[1] * height / diff_with_rois.shape[0]), height))
    
    # Mostrar imagens
    cv2.imshow("Frame Atual", frame_resized)
    cv2.imshow("Background", bg_resized)
    cv2.imshow("Diferença", diff_resized)
    
    print("\n=== INSTRUÇÕES ===")
    print("Pressione qualquer tecla para continuar...")
    print("ESC para sair")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def suggest_improvements():
    """
    Sugere melhorias baseadas na análise.
    """
    print("\n=== SUGESTÕES DE MELHORIAS ===")
    print("1. Ajustar OCCUPANCY_THRESHOLD baseado nos valores observados")
    print("2. Usar análise de textura além da diferença de pixels")
    print("3. Aplicar filtro de mediana para reduzir ruído")
    print("4. Usar múltiplos frames para confirmar ocupação")
    print("5. Implementar calibração automática de threshold")
    print("6. Considerar iluminação ambiente")


if __name__ == "__main__":
    analyze_parking_detection()
    suggest_improvements() 