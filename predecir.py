"""Localiza y reconoce señales en una fotografía sin volver a entrenar."""

import argparse
import json

import cv2
import numpy as np

from vision.clasificador_senales import ClasificadorSenales, MODELO_PREDETERMINADO
from vision.imagenes_cnn import leer_rgb
from vision.propuestas_senales import proponer_senales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imagen")
    parser.add_argument("--modelo", default=str(MODELO_PREDETERMINADO))
    parser.add_argument("--recorte", action="store_true", help="La imagen ya es un recorte; omite localización")
    args = parser.parse_args()
    try:
        clasificador = ClasificadorSenales(args.modelo)
        rgb = leer_rgb(args.imagen)
        if args.recorte:
            resultados = [clasificador.predecir(rgb)]
        else:
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            resultados = [{**clasificador.predecir(c["recorte"], "BGR"), "bounding_rect": c["bounding_rect"]}
                          for c in proponer_senales(bgr)]
        print(json.dumps({"detecciones": resultados}, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
