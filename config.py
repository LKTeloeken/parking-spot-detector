"""
Configurações de ROIs e parâmetros do detector.
Defina aqui as coordenadas (x, y, w, h) de cada vaga.
"""

# Exemplo de ROIs: cada tupla é (x, y, largura, altura)
PARKING_SPOTS = [
    (100,  50, 60, 120),
    (200,  50, 60, 120),
    # ... adicione todas as vagas manualmente após desenhar no primeiro frame
]

# Threshold para diferenciar vaga livre x ocupada (mudança de pixels)
OCCUPANCY_THRESHOLD = 500

# Espaço de cor para extração de cor dominante
COLOR_SPACE = 'HSV'