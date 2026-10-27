"""Demostración reproducible: fotos reales, línea y reloj simulados, CSV de pulsos."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from comunicacion.salida import SalidaRegistro
from control.contratos import ResultadoLinea
from control.estados import MaquinaEstados
from control.pulsos import ProgramadorPulsos
from vision.catalogo import CLASES
from vision.clasificador_senales import ClasificadorSenales
from vision.imagenes_cnn import leer_rgb
from vision.senales_cnn import DetectorSenalesCNN

RAIZ = Path(__file__).resolve().parents[1]


def demostrar():
    """Compone escenas a partir de validación y registra decisiones sin Bluetooth."""
    clasificador = ClasificadorSenales()
    detector = DetectorSenalesCNN(clasificador)
    with (RAIZ / "datos_preparados/particiones.csv").open(encoding="utf-8") as archivo:
        filas = [f for f in csv.DictReader(archivo) if f["conjunto"] == "val"]
    escenas, seleccion = {}, {}
    for indice, nombre in enumerate(CLASES):
        for fila in (f for f in filas if int(f["clase"]) == indice):
            rgb = leer_rgb(RAIZ / fila["ruta"]).astype(np.uint8)
            escena = np.full((480, 640, 3), 235, np.uint8)
            escena[:, 310:330] = 20  # Línea de fondo; el resultado de línea se simula abajo.
            escena[310:460, 245:395] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (150, 150))
            # Probar también la localización del recorte compuesto, no solo la CNN aislada.
            resultado = detector.detectar(escena)
            if any(c["color"] == nombre and c["confianza"] >= 0.8 for c in resultado.candidatos):
                escenas[nombre] = escena
                seleccion[nombre] = fila["ruta"]
                break
        if nombre not in escenas:
            raise RuntimeError(f"No se encontró una escena de demostración confirmable para {nombre}")
    carpeta = RAIZ / "resultados/demostracion"
    carpeta.mkdir(parents=True, exist_ok=True)
    salida = SalidaRegistro(str(carpeta / "comandos.csv"))
    maquina, pulsos = MaquinaEstados(), ProgramadorPulsos()
    linea = ResultadoLinea(valida=True, confianza=4)
    fases = [("SIGA", 4), ("VELOCIDAD_60", 4), ("ZONA_ESCOLAR", 6), ("PARE", 5), (None, 4)]
    detector.historial.clear()
    ticks = 0
    eventos = []
    pulsos_simulados = []
    try:
        for nombre, duracion in fases:
            escena = escenas[nombre] if nombre else np.full((480, 640, 3), 235, np.uint8)
            for _ in range(5):
                resultado = detector.detectar(escena)
            if nombre and (resultado.senal != nombre or not resultado.en_disparo):
                raise RuntimeError(f"La escena no confirmó {nombre}")
            eventos.append({"tiempo_simulado": ticks / 20, "senal": resultado.senal, "confianza": resultado.confianza})
            Image.fromarray(cv2.cvtColor(escena, cv2.COLOR_BGR2RGB)).save(carpeta / f"escena_{nombre or 'SIN_SENAL'}.jpg")
            for _ in range(duracion * 20):
                ahora = ticks / 20
                comando = maquina.actualizar(linea, resultado, ahora)
                if pulsos.permitir(comando, ahora, comando.intervalo_avance):
                    salida.enviar(comando)
                    pulsos_simulados.append({"tiempo_simulado": ahora, "accion": comando.accion,
                        "estado": comando.estado, "senal": comando.senal, "intervalo": comando.intervalo_avance})
                ticks += 1
    finally:
        salida.detener()
        salida.cerrar()
    with (carpeta / "tiempo_simulado.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(pulsos_simulados[0]))
        escritor.writeheader()
        escritor.writerows(pulsos_simulados)
    resumen = {"tipo": "Fotos de validación reales; escenas compuestas, línea y tiempo simulados. Sin hardware.",
               "fotos": seleccion, "fases": eventos, "cambios_estado": maquina.obtener_eventos(),
               "nota_tiempo": "La columna tiempo del CSV es monótona real; tiempo_simulado.csv indica el instante de cada pulso."}
    (carpeta / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demostrar()
