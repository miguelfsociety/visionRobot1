"""Comprueba los pesos reales exportados; se omite si aún no se entrenaron."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "resultados/senales_1"
MODELOS = ("VGG16", "VGG19", "MobileNet", "ResNet50", "DenseNet121")


@unittest.skipUnless(all((CARPETA / m / f"{m}.keras").exists() for m in MODELOS), "Entrena primero las cinco CNN")
class PruebasModelosReales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vision.clasificador_senales import ClasificadorSenales
        cls.clasificador = ClasificadorSenales(CARPETA / "MobileNet/MobileNet.keras")

    def test_exportaciones_coinciden_con_predicciones_del_entrenamiento(self):
        import keras
        from vision.clasificador_senales import ClasificadorSenales
        from vision.catalogo import CLASES
        for nombre in MODELOS:
            with self.subTest(modelo=nombre):
                clasificador = ClasificadorSenales(CARPETA / nombre / f"{nombre}.keras")
                with (CARPETA / nombre / "predicciones_test.csv").open(encoding="utf-8") as archivo:
                    filas = list(csv.DictReader(archivo))
                # Una muestra por clase: valida serialización, orden y preprocesamiento.
                for clase in range(4):
                    fila = next(f for f in filas if int(f["real"]) == clase)
                    resultado = clasificador.predecir(RAIZ / fila["ruta"])
                    np.testing.assert_allclose(list(resultado["probabilidades"].values()),
                        [float(fila[c]) for c in CLASES], atol=3e-4)
                del clasificador
                keras.backend.clear_session()

    def test_archivo_rgb_bgr_y_una_sola_carga(self):
        from vision.imagenes_cnn import leer_rgb
        with (CARPETA / "MobileNet/predicciones_test.csv").open(encoding="utf-8") as archivo:
            fila = next(csv.DictReader(archivo))
        ruta = RAIZ / fila["ruta"]
        rgb = leer_rgb(ruta)
        with patch("keras.models.load_model", side_effect=AssertionError("No debe recargar pesos")):
            resultados = [self.clasificador.predecir(ruta), self.clasificador.predecir(rgb),
                          self.clasificador.predecir(rgb[..., ::-1], "BGR")]
        np.testing.assert_allclose(list(resultados[0]["probabilidades"].values()), list(resultados[1]["probabilidades"].values()))
        np.testing.assert_allclose(list(resultados[0]["probabilidades"].values()), list(resultados[2]["probabilidades"].values()))
        with self.assertRaises(ValueError):
            self.clasificador.predecir(np.zeros((20, 20)))
        with self.assertRaises(ValueError):
            self.clasificador.predecir(rgb, "HSV")

    def test_modelo_incompatible_se_rechaza(self):
        import keras
        from vision.catalogo import CLASES
        from vision.clasificador_senales import ClasificadorSenales
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "digitos.keras"
            entrada = keras.Input((224, 224, 3))
            salida = keras.layers.Dense(10, activation="softmax")(keras.layers.GlobalAveragePooling2D()(entrada))
            keras.Model(entrada, salida).save(ruta)
            ruta.with_name("metadatos.json").write_text(json.dumps({"arquitectura": "MobileNet", "clases": list(CLASES),
                "tamano": [224, 224], "preprocesamiento": "MobileNet"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cuatro salidas"):
                ClasificadorSenales(ruta)
