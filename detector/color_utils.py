import cv2
import numpy as np

def get_dominant_color(image, k=3):
    """
    Retorna a cor dominante no ROI pelo método de k-means.
    """
    # reshape e converte
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # critérios e execução do k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())

    # seleciona a cor mais frequente
    dominant = centers[np.argmax(counts)]
    return tuple(map(int, dominant))