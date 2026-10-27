"""Lectura RGB y resize compartidos por entrenamiento e inferencia."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def leer_rgb(imagen, orden_color="RGB"):
    """Recibe ruta o matriz H×W×3 en [0,255] y devuelve RGB float32."""
    if orden_color not in {"RGB", "BGR"}:
        raise ValueError("orden_color debe ser RGB o BGR")
    if isinstance(imagen, (str, Path)):
        if Path(imagen).suffix.lower() in {".heic", ".heif"}:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        with Image.open(imagen) as foto:
            datos = np.asarray(ImageOps.exif_transpose(foto).convert("RGB"), dtype=np.float32)
    else:
        datos = np.asarray(imagen, dtype=np.float32)
        if orden_color == "BGR":
            datos = datos[..., ::-1]
    if datos.ndim != 3 or datos.shape[-1] != 3 or not all(datos.shape[:2]):
        raise ValueError("La imagen debe tener alto, ancho y tres canales")
    if not np.isfinite(datos).all() or datos.min() < 0 or datos.max() > 255:
        raise ValueError("Los píxeles deben ser finitos y estar en [0,255]")
    return np.ascontiguousarray(datos)


def preparar_lote(imagen, tamano, preprocesar, orden_color="RGB"):
    """Complejidad: O(alto × ancho). Devuelve un lote preprocesado."""
    rgb = leer_rgb(imagen, orden_color)
    return preprocesar(tf.expand_dims(tf.image.resize(rgb, tamano), 0))
