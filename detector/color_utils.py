import cv2
import numpy as np
import webcolors 

def get_dominant_color(image: np.ndarray, mask: np.ndarray = None, k: int = 3):
    """
    Retorna a cor dominante no ROI pelo método de k-means,
    opcionalmente usando apenas os pixels onde mask > 0.
    image: BGR
    mask: 2D uint8, com 0/255, mesmo tamanho de image[:,:,0]
    """
    # converte de BGR para RGB para formatar o hex depois
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # obtém apenas pixels onde mask==255, senão usa tudo
    if mask is not None:
        flat_mask = mask.flatten()
        pixels = img_rgb.reshape(-1, 3)[flat_mask > 0]
        if pixels.size == 0:
            return None
        data = np.float32(pixels)
    else:
        data = img_rgb.reshape(-1, 3).astype(np.float32)

    # k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # conta qual centro é mais frequente
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]

    # retorna como tupla de int RGB
    return tuple(int(c) for c in dominant)

def get_color_name(requested_colour):
    """
    Tenta retornar o nome CSS3 exato; 
    se não encontrar, faz busca pelo nome mais próximo.
    rgb_tuple: (R, G, B)
    """
    min_dist = float("inf")
    closest_name = None

    # get list of all CSS3 names (lowercase)
    for name in webcolors.names("css3"):
        # map name → RGB tuple
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        # Euclidean distance in RGB space
        dist = (r_c - requested_colour[0])**2 + \
               (g_c - requested_colour[1])**2 + \
               (b_c - requested_colour[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name
    


# def closest_colour(requested_colour):
#     min_dist = float("inf")
#     closest_name = None

#     # get list of all CSS3 names (lowercase)
#     for name in webcolors.names("css3"):
#         # map name → RGB tuple
#         r_c, g_c, b_c = webcolors.name_to_rgb(name)
#         # Euclidean distance in RGB space
#         dist = (r_c - requested_colour[0])**2 + \
#                (g_c - requested_colour[1])**2 + \
#                (b_c - requested_colour[2])**2
#         if dist < min_dist:
#             min_dist = dist
#             closest_name = name

#     return closest_name