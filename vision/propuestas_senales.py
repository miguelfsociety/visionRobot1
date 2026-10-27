"""Localiza regiones de color sin asignarles una clase de tránsito.

Recibe BGR y devuelve recortes BGR y coordenadas en la imagen recibida.
Complejidad: O(alto × ancho), con un máximo configurable de candidatos.
"""

import cv2
import numpy as np

import config


def proponer_senales(imagen_bgr, cfg=config):
    """Devuelve propuestas ordenadas por área; el color no decide la clase."""
    if imagen_bgr is None or imagen_bgr.ndim != 3 or imagen_bgr.shape[2] != 3:
        raise ValueError("Se esperaba una imagen BGR de tres canales.")
    alto, ancho = imagen_bgr.shape[:2]
    escala = min(1.0, cfg.CNN_ANCHO_PROPUESTAS / ancho)
    reducida = cv2.resize(imagen_bgr, (max(1, round(ancho * escala)), max(1, round(alto * escala))))
    hsv = cv2.cvtColor(reducida, cv2.COLOR_BGR2HSV)
    kernel = np.ones((cfg.CNN_KERNEL, cfg.CNN_KERNEL), np.uint8)
    candidatos = []
    for color, rangos in cfg.CNN_RANGOS_HSV.items():
        mascara = np.zeros(hsv.shape[:2], np.uint8)
        for inferior, superior in rangos:
            mascara = cv2.bitwise_or(mascara, cv2.inRange(hsv, np.array(inferior), np.array(superior)))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            fraccion = area / mascara.size
            x, y, w, h = cv2.boundingRect(contorno)
            if not cfg.CNN_AREA_MINIMA <= fraccion <= cfg.CNN_AREA_MAXIMA:
                continue
            if not cfg.CNN_ASPECTO_MINIMO <= w / h <= cfg.CNN_ASPECTO_MAXIMO:
                continue
            if area / (w * h) < cfg.CNN_EXTENSION_MINIMA:
                continue
            x, y, w, h = (round(v / escala) for v in (x, y, w, h))
            margen = round(max(w, h) * cfg.CNN_MARGEN_RECORTE)
            x0, y0 = max(0, x - margen), max(0, y - margen)
            x1, y1 = min(ancho, x + w + margen), min(alto, y + h + margen)
            candidatos.append({
                "color_propuesta": color, "color": color,
                "area": area / escala ** 2,
                "bounding_rect": (x0, y0, x1 - x0, y1 - y0),
                "centroide": (x + w / 2, y + h / 2),
                "recorte": imagen_bgr[y0:y1, x0:x1].copy(),
            })
    return sorted(candidatos, key=lambda c: c["area"], reverse=True)[:cfg.CNN_MAX_CANDIDATOS]
