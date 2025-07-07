"""
Configurações de ROIs e parâmetros do detector.
"""

# Exemplo de ROIs: cada tupla é (x, y, largura, altura)
PARKING_SPOTS = [
    (10, 140, 125, 300),
    (200, 140, 200, 300),
    (450, 140, 200, 300),
    (700, 140, 150, 285),
]

# Threshold para diferenciar vaga livre x ocupada (mudança de pixels)
OCCUPANCY_THRESHOLD = 500

# Espaço de cor para extração de cor dominante
COLOR_SPACE = "HSV"

