import cv2
import numpy as np
from config_diagonal import PARKING_SPOTS_CUSTOM, POLYGON_OCCUPANCY_THRESHOLD
from detector.color_utils import get_dominant_color


class PolygonParkingDetector:
    def __init__(self, polygons=None):
        self.spots = polygons if polygons is not None else PARKING_SPOTS_CUSTOM
        self.spot_masks = {}
        self.spot_bounding_boxes = {}
        self._prepare_masks()
        
    def _prepare_masks(self):
        """
        Prepara as máscaras e bounding boxes para cada vaga.
        """
        for idx, polygon in enumerate(self.spots):
            # Converter para numpy array se necessário
            if isinstance(polygon, list):
                polygon = np.array(polygon)
            
            # Calcular bounding box
            x_min = int(np.min(polygon[:, 0]))
            y_min = int(np.min(polygon[:, 1]))
            x_max = int(np.max(polygon[:, 0]))
            y_max = int(np.max(polygon[:, 1]))
            
            self.spot_bounding_boxes[idx] = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Criar máscara para o polígono
            mask = np.zeros((y_max - y_min + 50, x_max - x_min + 50), dtype=np.uint8)
            
            # Ajustar coordenadas do polígono para a máscara local
            local_polygon = polygon.copy()
            local_polygon[:, 0] -= x_min
            local_polygon[:, 1] -= y_min
            
            # Preencher polígono na máscara
            cv2.fillPoly(mask, [local_polygon.astype(np.int32)], 255)
            
            self.spot_masks[idx] = mask
    
    def _extract_polygon_roi(self, image, polygon_idx):
        """
        Extrai a ROI usando a máscara do polígono.
        """
        x, y, w, h = self.spot_bounding_boxes[polygon_idx]
        mask = self.spot_masks[polygon_idx]
        
        # Extrair região da imagem
        roi = image[y:y+h, x:x+w]
        
        # Aplicar máscara
        if roi.shape[:2] != mask.shape[:2]:
            # Redimensionar máscara se necessário
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
        
        # Aplicar máscara
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        return masked_roi, mask
    
    def detect(self, frame: np.ndarray, bg_frame: np.ndarray = None) -> list:
        """
        Detecta ocupação usando polígonos.
        """
        results = []
        
        if bg_frame is not None:
            return self._detect_with_background(frame, bg_frame)
        else:
            return self._detect_simple(frame)
    
    def _detect_with_background(self, frame: np.ndarray, bg_frame: np.ndarray) -> list:
        """
        Detecção usando frame de background.
        """
        results = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        
        for idx in range(len(self.spots)):
            # Extrair ROIs
            roi_frame, mask = self._extract_polygon_roi(frame_gray, idx)
            roi_bg, _ = self._extract_polygon_roi(bg_gray, idx)
            
            # Calcular diferença
            diff = cv2.absdiff(roi_bg, roi_frame)
            
            # Aplicar máscara na diferença
            diff_masked = cv2.bitwise_and(diff, diff, mask=mask)
            
            # Contar pixels diferentes
            non_zero_pixels = cv2.countNonZero(diff_masked)
            
            # Determinar ocupação (usando porcentagem da área)
            total_pixels = cv2.countNonZero(mask)
            occupied = (non_zero_pixels / total_pixels) >= POLYGON_OCCUPANCY_THRESHOLD if total_pixels > 0 else False
            
            # Extrair cor dominante se ocupado
            color = None
            if occupied:
                x, y, w, h = self.spot_bounding_boxes[idx]
                spot_img = frame[y:y+h, x:x+w]
                
                # Aplicar máscara colorida
                if len(spot_img.shape) == 3:
                    mask_3d = cv2.merge([mask, mask, mask])
                    spot_img_masked = cv2.bitwise_and(spot_img, spot_img, mask=mask)
                    
                    # Extrair cor apenas da área mascarada
                    non_zero_pixels_color = spot_img_masked[mask > 0]
                    if len(non_zero_pixels_color) > 0:
                        color = tuple(map(int, np.mean(non_zero_pixels_color, axis=0)))
            
            results.append((occupied, color))
        
        return results
    
    def _detect_simple(self, frame: np.ndarray) -> list:
        """
        Detecção simples sem background.
        """
        results = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for idx in range(len(self.spots)):
            roi, mask = self._extract_polygon_roi(frame_gray, idx)
            
            # Aplicar máscara
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
            
            # Análise de textura
            variance = np.var(roi_masked[mask > 0])
            mean_intensity = np.mean(roi_masked[mask > 0])
            
            # Heurística para detecção
            occupied = variance > 300 and (mean_intensity < 60 or mean_intensity > 120)
            
            # Extrair cor se ocupado
            color = None
            if occupied:
                x, y, w, h = self.spot_bounding_boxes[idx]
                spot_img = frame[y:y+h, x:x+w]
                
                if len(spot_img.shape) == 3:
                    spot_img_masked = cv2.bitwise_and(spot_img, spot_img, mask=mask)
                    non_zero_pixels_color = spot_img_masked[mask > 0]
                    
                    if len(non_zero_pixels_color) > 0:
                        color = tuple(map(int, np.mean(non_zero_pixels_color, axis=0)))
            
            results.append((occupied, color))
        
        return results
    
    def draw_annotations(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Desenha anotações com polígonos.
        """
        annotated = frame.copy()
        
        for idx, (polygon, (occupied, color)) in enumerate(zip(self.spots, detections)):
            # Converter para numpy array se necessário
            if isinstance(polygon, list):
                polygon = np.array(polygon)
            
            # Cor do polígono
            poly_color = (0, 0, 255) if occupied else (0, 255, 0)
            
            # Desenhar polígono
            cv2.polylines(annotated, [polygon.astype(np.int32)], True, poly_color, 2)
            
            # Preencher polígono com transparência
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon.astype(np.int32)], poly_color)
            cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0, annotated)
            
            # Adicionar label
            center_x = int(np.mean(polygon[:, 0]))
            center_y = int(np.mean(polygon[:, 1]))
            
            label = f"V{idx+1}: {'OCUPADA' if occupied else 'LIVRE'}"
            cv2.putText(annotated, label, (center_x - 50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Círculo da cor dominante
            if color:
                cv2.circle(annotated, (center_x, center_y - 20), 10, color, -1)
        
        return annotated
    
    def get_interactive_selector(self, frame: np.ndarray):
        """
        Permite seleção interativa de vagas na diagonal.
        """
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                param['points'].append([x, y])
                print(f"Ponto adicionado: ({x}, {y})")
                
                # Desenhar ponto
                cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Seletor de Vagas', param['image'])
                
                # Se tem 4 pontos, completar polígono
                if len(param['points']) == 4:
                    polygon = np.array(param['points'])
                    cv2.polylines(param['image'], [polygon], True, (255, 0, 0), 2)
                    cv2.imshow('Seletor de Vagas', param['image'])
                    
                    param['polygons'].append(polygon)
                    param['points'] = []
                    print(f"Polígono {len(param['polygons'])} criado!")
        
        # Parâmetros para callback
        params = {
            'image': frame.copy(),
            'points': [],
            'polygons': []
        }
        
        cv2.namedWindow('Seletor de Vagas')
        cv2.setMouseCallback('Seletor de Vagas', mouse_callback, params)
        cv2.imshow('Seletor de Vagas', params['image'])
        
        print("=== SELETOR INTERATIVO DE VAGAS ===")
        print("Clique em 4 pontos para definir cada vaga (no sentido horário)")
        print("Pressione 's' para salvar, 'r' para resetar, 'q' para sair")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_polygons(params['polygons'])
                print("Polígonos salvos!")
            elif key == ord('r'):
                params['polygons'] = []
                params['points'] = []
                params['image'] = frame.copy()
                cv2.imshow('Seletor de Vagas', params['image'])
                print("Resetado!")
        
        cv2.destroyAllWindows()
        return params['polygons']
    
    def _save_polygons(self, polygons):
        """
        Salva os polígonos em um arquivo de configuração.
        """
        config_text = "# Configuração gerada automaticamente\n"
        config_text += "import numpy as np\n\n"
        config_text += "PARKING_SPOTS_CUSTOM = [\n"
        
        for i, polygon in enumerate(polygons):
            config_text += f"    # Vaga {i+1}\n"
            config_text += "    np.array([\n"
            for point in polygon:
                config_text += f"        [{point[0]}, {point[1]}],\n"
            config_text += "    ]),\n\n"
        
        config_text += "]\n"
        
        with open('config_custom_polygons.py', 'w') as f:
            f.write(config_text)
        
        print("Configuração salva em 'config_custom_polygons.py'") 