"""Reevalúa un modelo guardado sobre la partición de prueba persistente."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from vision.catalogo import CLASES
from vision.clasificador_senales import ClasificadorSenales, MODELO_PREDETERMINADO

RAIZ = Path(__file__).resolve().parents[1]


def evaluar(modelo, particiones):
    """Carga el modelo completo y calcula métricas sin volver a entrenar."""
    clasificador = ClasificadorSenales(modelo)
    with Path(particiones).open(encoding="utf-8") as archivo:
        filas = [f for f in csv.DictReader(archivo) if f["conjunto"] == "test"]
    if not filas:
        raise ValueError("No se encontraron muestras de prueba")
    resultados = [clasificador.predecir(RAIZ / f["ruta"]) for f in filas]
    reporte = classification_report([int(f["clase"]) for f in filas], [r["identificador"] for r in resultados],
        labels=list(range(4)), target_names=CLASES, output_dict=True, zero_division=0)
    reporte["cobertura_confianza_080"] = float(np.mean([r["confianza"] >= 0.8 for r in resultados]))
    salida = Path(modelo).parent / "evaluacion_recargada.json"
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    return reporte


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=MODELO_PREDETERMINADO)
    parser.add_argument("--particiones", type=Path, default=RAIZ / "datos_preparados/particiones.csv")
    args = parser.parse_args()
    evaluar(args.modelo, args.particiones)
