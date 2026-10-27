"""Ejercita main.py con video compuesto y registro, sin cámara ni Bluetooth.

Ejecutar desde la raíz, después de pruebas.demostracion_cnn. Reproduce el
video a ritmo real (20 segundos) y calibra las ROI solo durante esta prueba.
"""

import csv
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np

import main
from vision.captura import FuenteVideo
from PIL import Image


def probar():
    carpeta = Path("resultados/demostracion")
    video = carpeta / "pipeline.avi"
    escritor = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
    if not escritor.isOpened():
        raise RuntimeError("No se pudo crear el video de prueba")
    try:
        for nombre in ("SIGA", "VELOCIDAD_60", "ZONA_ESCOLAR", "PARE", "SIN_SENAL"):
            with Image.open(carpeta / f"escena_{nombre}.jpg") as foto:
                imagen = cv2.cvtColor(np.asarray(foto.convert("RGB")), cv2.COLOR_RGB2BGR)
            for _ in range(80):
                escritor.write(imagen)
    finally:
        escritor.release()
    leer_original = FuenteVideo.leer
    def leer_pausado(self):
        time.sleep(0.05)
        return leer_original(self)
    args = main.construir_parser().parse_args(["--video", str(video), "--etapa", "completo", "--salida", "registro"])
    with patch.multiple(main.config, ROTACION=0, FILA_CHASIS=1.0, ROI_SENALES_FIN=1.0), \
            patch.object(FuenteVideo, "leer", leer_pausado):
        main.ejecutar(args)
    with Path("resultados/telemetria/metricas_fotogramas.csv").open(encoding="utf-8") as archivo:
        fotogramas = list(csv.DictReader(archivo))
    if len(fotogramas) != 400:
        raise AssertionError(f"Se esperaban 400 fotogramas; llegaron {len(fotogramas)}")
    with Path("resultados/comandos.csv").open(encoding="utf-8") as archivo:
        comandos = list(csv.DictReader(archivo))
    observadas = {c["senal"] for c in comandos}
    if not {"SIGA", "VELOCIDAD_60", "ZONA_ESCOLAR", "PARE"}.issubset(observadas):
        raise AssertionError(f"Faltan señales en el flujo principal: {observadas}")
    print(f"Pipeline validado: {len(fotogramas)} fotogramas y {len(comandos)} comandos registrados. Sin hardware.")


if __name__ == "__main__":
    probar()
