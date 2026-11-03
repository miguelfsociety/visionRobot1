"""Audita propuestas y CNN sobre la zona de cámara de una grabación del debug.

No genera etiquetas de entrenamiento ni mide exactitud sin anotaciones humanas.
La zona de cámara se especifica en píxeles de la grabación de pantalla.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import tensorflow as tf

import config
from vision.clasificador_senales import ClasificadorSenales
from vision.propuestas_senales import proponer_senales


def analizar(video, modelos, zona, paso, salida):
    """Compara modelos sobre los mismos candidatos antes/después del filtro."""
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    clasificadores = {str(m): ClasificadorSenales(m) for m in modelos}
    cfg_anterior = SimpleNamespace(**vars(config))
    cfg_anterior.CNN_VERTICES_MINIMOS = 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir {video}")
    fps, cantidad = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    muestras = []
    try:
        for indice in range(0, int(cantidad), max(1, round(fps * paso))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, cuadro = cap.read()
            if not ok:
                break
            x0, y0, x1, y1 = zona
            camara = cuadro[y0:y1, x0:x1]
            alto = round(camara.shape[0] * config.ANCHO_PROCESO / camara.shape[1])
            camara = cv2.resize(camara, (config.ANCHO_PROCESO, alto))
            inicio = round(alto * config.FILA_CHASIS * config.ROI_SENALES_INICIO)
            fin = round(alto * config.FILA_CHASIS * config.ROI_SENALES_FIN)
            roi = camara[inicio:fin]
            propuestas = proponer_senales(roi, cfg_anterior)
            aceptadas = {p["bounding_rect"] for p in proponer_senales(roi)}
            candidatos = []
            for propuesta in propuestas:
                predicciones = {nombre: c.predecir(propuesta["recorte"], "BGR") for nombre, c in clasificadores.items()}
                candidatos.append({"bounding_rect": propuesta["bounding_rect"],
                    "aceptado_filtro_nuevo": propuesta["bounding_rect"] in aceptadas,
                    "en_disparo": propuesta["centroide"][1] >= config.Y_DISPARO * roi.shape[0],
                    "predicciones": predicciones})
            muestras.append({"segundo": indice / fps, "candidatos": candidatos})
    finally:
        cap.release()
    informe = {"video": str(video), "zona_camara": zona, "paso_s": paso,
               "advertencia": "Grabación del debug con dibujos superpuestos; auditoría visual, no exactitud anotada ni datos para entrenar.",
               "muestras": muestras}
    Path(salida).write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Auditoría guardada: {salida}; {len(muestras)} muestras", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--modelos", nargs="+", required=True)
    parser.add_argument("--zona", nargs=4, type=int, required=True)
    parser.add_argument("--paso", type=float, default=1)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()
    analizar(args.video, args.modelos, args.zona, args.paso, args.salida)
