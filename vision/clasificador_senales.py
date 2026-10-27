"""Clase reutilizable de inferencia de las cuatro señales de tránsito."""

import json
from pathlib import Path

import keras
from keras import applications
import numpy as np

from vision.catalogo import CLASES
from vision.imagenes_cnn import preparar_lote

MODELO_PREDETERMINADO = Path(__file__).resolve().parents[1] / "resultados/senales_1/MobileNet/MobileNet.keras"
PREPROCESADORES = {
    "VGG16": applications.vgg16.preprocess_input,
    "VGG19": applications.vgg19.preprocess_input,
    "MobileNet": applications.mobilenet.preprocess_input,
    "ResNet50": applications.resnet50.preprocess_input,
    "DenseNet121": applications.densenet.preprocess_input,
}


class ClasificadorSenales:
    """Carga una CNN una vez; cada llamada clasifica un recorte de señal."""

    def __init__(self, ruta_modelo=MODELO_PREDETERMINADO):
        ruta = Path(ruta_modelo)
        self.metadatos = json.loads(ruta.with_name("metadatos.json").read_text(encoding="utf-8"))
        arquitectura = self.metadatos.get("arquitectura")
        if arquitectura not in PREPROCESADORES or self.metadatos.get("clases") != list(CLASES):
            raise ValueError("Los metadatos no corresponden a las cuatro señales esperadas")
        self.modelo = keras.models.load_model(ruta, compile=False)
        forma = self.modelo.input_shape
        if not isinstance(forma, tuple) or len(forma) != 4 or forma[3] != 3 or None in forma[1:3]:
            raise ValueError("El modelo debe recibir imágenes RGB de tamaño fijo")
        if self.modelo.output_shape != (None, 4):
            raise ValueError("El modelo debe tener cuatro salidas; no admite modelos de dígitos")
        if self.modelo.name != arquitectura:
            raise ValueError("La arquitectura del modelo no coincide con sus metadatos")
        if list(forma[1:3]) != self.metadatos.get("tamano") or self.metadatos.get("preprocesamiento") != arquitectura:
            raise ValueError("El tamaño/preprocesamiento del modelo no coincide con sus metadatos")
        self.tamano = forma[1:3]
        self.preprocesar = PREPROCESADORES[arquitectura]

    def predecir(self, imagen, orden_color="RGB"):
        """Recibe ruta o matriz y devuelve clase, nombre, confianza y probabilidades.

        Complejidad: O(alto × ancho) de lectura más una inferencia de la CNN.
        """
        lote = preparar_lote(imagen, self.tamano, self.preprocesar, orden_color)
        probabilidades = np.asarray(self.modelo(lote, training=False))[0]
        if (probabilidades.shape != (4,) or not np.isfinite(probabilidades).all()
                or np.any(probabilidades < 0) or np.any(probabilidades > 1)
                or not np.isclose(probabilidades.sum(), 1, atol=1e-5)):
            raise ValueError("La CNN no devolvió probabilidades softmax válidas")
        indice = int(probabilidades.argmax())
        return {"identificador": indice, "nombre": CLASES[indice], "confianza": float(probabilidades[indice]),
                "probabilidades": {nombre: float(p) for nombre, p in zip(CLASES, probabilidades)}}
