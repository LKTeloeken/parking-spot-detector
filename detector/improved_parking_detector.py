import cv2
import numpy as np
from config import PARKING_SPOTS
from detector.color_utils import get_dominant_color


class ImprovedParkingDetector:
    def __init__(self):
        self.spots = PARKING_SPOTS
        self.adaptive_thresholds = {}
        self.calibrated = False
        
    def calibrate_thresholds(self, bg_frame: np.ndarray, sample_frames: list):
        """
        Calibra os thresholds automaticamente baseado em frames de amostra.
        """
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        
        for idx, (x, y, w, h) in enumerate(self.spots):
            differences = []
            
            # Analisa diferenças em múltiplos frames
            for frame in sample_frames:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Aplicar filtro de mediana para reduzir ruído
                roi_frame = cv2.medianBlur(frame_gray[y:y+h, x:x+w], 5)
                roi_bg = cv2.medianBlur(bg_gray[y:y+h, x:x+w], 5)
                
                # Calcular diferença
                diff = cv2.absdiff(roi_bg, roi_frame)
                non_zero = cv2.countNonZero(diff)
                differences.append(non_zero)
            
            # Usar percentil 90 como threshold base
            base_threshold = np.percentile(differences, 90)
            # Ajustar baseado no tamanho da vaga
            area = w * h
            adaptive_threshold = base_threshold + (area * 0.1)  # 10% do tamanho da área
            
            self.adaptive_thresholds[idx] = adaptive_threshold
            
        self.calibrated = True
        print("Thresholds calibrados:")
        for idx, threshold in self.adaptive_thresholds.items():
            print(f"  Vaga {idx + 1}: {threshold:.0f}")

    def detect_with_texture_analysis(self, frame: np.ndarray, bg_frame: np.ndarray) -> list:
        """
        Detecção melhorada usando análise de textura e múltiplos critérios.
        """
        results = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        
        for idx, (x, y, w, h) in enumerate(self.spots):
            # Aplicar filtro de mediana para reduzir ruído
            roi_frame = cv2.medianBlur(frame_gray[y:y+h, x:x+w], 5)
            roi_bg = cv2.medianBlur(bg_gray[y:y+h, x:x+w], 5)
            
            # Critério 1: Diferença de pixels
            diff = cv2.absdiff(roi_bg, roi_frame)
            non_zero_pixels = cv2.countNonZero(diff)
            
            # Critério 2: Análise de variância (textura)
            variance_frame = np.var(roi_frame)
            variance_bg = np.var(roi_bg)
            texture_diff = abs(variance_frame - variance_bg)
            
            # Critério 3: Análise de gradiente
            grad_x = cv2.Sobel(roi_frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi_frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # Critério 4: Análise de histograma
            hist_frame = cv2.calcHist([roi_frame], [0], None, [256], [0, 256])
            hist_bg = cv2.calcHist([roi_bg], [0], None, [256], [0, 256])
            hist_correlation = cv2.compareHist(hist_frame, hist_bg, cv2.HISTCMP_CORREL)
            
            # Decisão baseada em múltiplos critérios
            occupied = self._make_decision(idx, non_zero_pixels, texture_diff, 
                                         gradient_mean, hist_correlation)
            
            color = None
            if occupied:
                spot_img = frame[y:y+h, x:x+w]
                color = get_dominant_color(spot_img)
                
            results.append((occupied, color))
            
        return results
    
    def _make_decision(self, spot_idx: int, pixel_diff: int, texture_diff: float, 
                      gradient_mean: float, hist_correlation: float) -> bool:
        """
        Toma decisão baseada em múltiplos critérios.
        """
        # Usar threshold adaptativo se calibrado
        if self.calibrated and spot_idx in self.adaptive_thresholds:
            threshold = self.adaptive_thresholds[spot_idx]
        else:
            # Fallback para threshold baseado no tamanho da vaga
            x, y, w, h = self.spots[spot_idx]
            area = w * h
            threshold = max(5000, area * 0.15)  # Mínimo 5000 ou 15% da área
        
        # Critério principal: diferença de pixels
        pixel_criteria = pixel_diff > threshold
        
        # Critério de textura: mudança significativa na variância
        texture_criteria = texture_diff > 50
        
        # Critério de gradiente: presença de bordas/objetos
        gradient_criteria = gradient_mean > 10
        
        # Critério de histograma: baixa correlação indica mudança
        hist_criteria = hist_correlation < 0.7
        
        # Decisão: pelo menos 2 critérios devem ser atendidos
        criteria_met = sum([pixel_criteria, texture_criteria, 
                           gradient_criteria, hist_criteria])
        
        return criteria_met >= 2
    
    def detect(self, frame: np.ndarray, bg_frame: np.ndarray = None) -> list:
        """
        Método principal de detecção.
        """
        if bg_frame is not None:
            return self.detect_with_texture_analysis(frame, bg_frame)
        else:
            # Fallback para detecção simples
            return self._simple_detect(frame)
    
    def _simple_detect(self, frame: np.ndarray) -> list:
        """
        Detecção simples sem frame de background.
        """
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for idx, (x, y, w, h) in enumerate(self.spots):
            roi = gray[y:y+h, x:x+w]
            
            # Usar análise de textura para detectar objetos
            variance = np.var(roi)
            mean_intensity = np.mean(roi)
            
            # Heurística: áreas com carros tendem a ter mais variância
            # e intensidade diferente do asfalto
            occupied = variance > 300 and (mean_intensity < 60 or mean_intensity > 120)
            
            color = None
            if occupied:
                spot_img = frame[y:y+h, x:x+w]
                color = get_dominant_color(spot_img)
                
            results.append((occupied, color))
            
        return results
    
    def draw_annotations(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Desenha anotações no frame com informações detalhadas.
        """
        annotated = frame.copy()
        
        for idx, ((x, y, w, h), (occupied, color)) in enumerate(zip(self.spots, detections)):
            # Cor do retângulo
            rect_color = (0, 0, 255) if occupied else (0, 255, 0)
            
            # Desenhar retângulo
            cv2.rectangle(annotated, (x, y), (x+w, y+h), rect_color, 2)
            
            # Label
            label = f"V{idx+1}: {'OCUPADA' if occupied else 'LIVRE'}"
            cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Mostrar threshold se calibrado
            if self.calibrated and idx in self.adaptive_thresholds:
                threshold_text = f"T: {self.adaptive_thresholds[idx]:.0f}"
                cv2.putText(annotated, threshold_text, (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Círculo da cor dominante
            if color:
                cv2.circle(annotated, (x + 15, y + 15), 10, color, -1)
                
        return annotated 