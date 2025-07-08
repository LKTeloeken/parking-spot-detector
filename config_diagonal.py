"""
Configurações para vagas na diagonal usando polígonos.
"""

# Configurações para detector com polígonos personalizados
# Cada vaga é definida por uma lista de 4 pontos (x, y)

# Layout personalizado final - formato específico para cada vaga
PARKING_SPOTS_CUSTOM = [
    # V1: Formato trapézio - topo maior que base
    [[5, 140], [170, 160], [155, 440], [15, 440]],
    
    # V2: Largura reduzida - mais estreito
    [[220, 150], [360, 140], [360, 440], [200, 440]],
    
    # V3: Diagonal para direita
    [[450, 140], [660, 150], [680, 440], [450, 440]],
    
    # V4: Diagonal para direita
    [[700, 140], [860, 150], [870, 425], [700, 425]]
]

# Configurações de detecção
POLYGON_OCCUPANCY_THRESHOLD = 0.15  # 15% de diferença para considerar ocupado
POLYGON_GAUSSIAN_BLUR = (5, 5)      # Suavização para reduzir ruído
POLYGON_EDGE_THRESHOLD = 30         # Threshold para detecção de bordas 