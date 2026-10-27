"""Copia las cuatro clases, convierte HEIC, deduplica y prepara recortes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split

from vision.catalogo import CLASES
from vision.propuestas_senales import proponer_senales

RAIZ = Path(__file__).resolve().parents[1]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".webp"}


def preparar(origen, destino=RAIZ, semilla=42):
    """Prepara datos reproducibles sin modificar las fotos de origen."""
    register_heif_opener()
    origen, destino = Path(origen).resolve(), Path(destino).resolve()
    if origen == destino / "dataset":
        raise ValueError("El origen debe ser distinto del dataset de destino.")
    manifiesto, aceptados, vistos, pixeles_vistos = [], [], {}, {}
    for indice, nombre in enumerate(CLASES):
        carpeta = origen / str(indice)
        if not carpeta.is_dir():
            raise ValueError(f"No existe la carpeta {carpeta}")
        (destino / "dataset" / str(indice)).mkdir(parents=True, exist_ok=True)
        (destino / "datos_preparados" / str(indice)).mkdir(parents=True, exist_ok=True)
        for foto in sorted(p for p in carpeta.rglob("*") if p.is_file()):
            registro = {"origen": str(foto), "clase": indice, "senal": nombre}
            manifiesto.append(registro)
            try:
                if foto.suffix.lower() not in EXTENSIONES:
                    raise ValueError("Extensión de imagen no admitida")
                huella = hashlib.sha256(foto.read_bytes()).hexdigest()
                registro["sha256"] = huella
                if huella in vistos:
                    if vistos[huella][0] != indice:
                        raise ValueError("El mismo archivo aparece en clases distintas")
                    registro.update(estado="duplicado", duplicado_de=vistos[huella][1])
                    continue
                vistos[huella] = (indice, str(foto))
                with Image.open(foto) as original:
                    rgb = ImageOps.exif_transpose(original).convert("RGB")
                    datos = np.asarray(rgb)
                    huella_pixeles = hashlib.sha256(datos.tobytes()).hexdigest()
                    if huella_pixeles in pixeles_vistos:
                        if pixeles_vistos[huella_pixeles][0] != indice:
                            raise ValueError("Píxeles idénticos con etiquetas distintas")
                        registro.update(estado="duplicado", duplicado_de=pixeles_vistos[huella_pixeles][1])
                        continue
                    pixeles_vistos[huella_pixeles] = (indice, str(foto))
                    nombre_salida = f"{foto.stem}_{huella[:10]}.jpg"
                    copia = destino / "dataset" / str(indice) / nombre_salida
                    rgb.save(copia, quality=95)
                propuestas = proponer_senales(cv2.cvtColor(datos, cv2.COLOR_RGB2BGR))
                registro.update(copia=str(copia.relative_to(destino)), convertido_heic=foto.suffix.lower() in {".heic", ".heif"})
                if not propuestas:
                    registro["estado"] = "sin_recorte"
                    continue
                propuesta = propuestas[0]
                recorte = destino / "datos_preparados" / str(indice) / nombre_salida
                Image.fromarray(cv2.cvtColor(propuesta["recorte"], cv2.COLOR_BGR2RGB)).save(recorte, quality=95)
                registro.update(estado="aceptado", recorte=str(recorte.relative_to(destino)),
                                bounding_rect=propuesta["bounding_rect"], sha256_pixeles=huella_pixeles)
                aceptados.append(registro)
            except (OSError, ValueError, cv2.error) as error:
                registro.update(estado="error", error=str(error))
    salida = destino / "datos_preparados"
    (salida / "manifiesto.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    rechazados = [r for r in manifiesto if r["estado"] not in {"aceptado", "duplicado"}]
    (salida / "rechazados.json").write_text(json.dumps(rechazados, ensure_ascii=False, indent=2), encoding="utf-8")
    indices = list(range(len(aceptados)))
    etiquetas = [r["clase"] for r in aceptados]
    resto, prueba = train_test_split(indices, test_size=0.1, stratify=etiquetas, random_state=semilla)
    entrenamiento, validacion = train_test_split(resto, test_size=0.2/0.9,
        stratify=[etiquetas[i] for i in resto], random_state=semilla)
    particiones = []
    for conjunto, seleccion in (("train", entrenamiento), ("val", validacion), ("test", prueba)):
        for i in sorted(seleccion):
            r = aceptados[i]
            particiones.append({"ruta": r["recorte"], "clase": r["clase"], "conjunto": conjunto,
                                "sha256": r["sha256"], "sha256_pixeles": r["sha256_pixeles"]})
    with (salida / "particiones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(particiones[0]))
        escritor.writeheader()
        escritor.writerows(particiones)
    resumen = {"archivos": len(manifiesto), "aceptados": len(aceptados), "duplicados": sum(r["estado"] == "duplicado" for r in manifiesto),
               "rechazados": len(rechazados), "heic_convertidos": sum(r.get("convertido_heic", False) for r in manifiesto),
               "clases": {str(i): n for i, n in enumerate(CLASES)}, "semilla": semilla,
               "particiones": {c: sum(r["conjunto"] == c for r in particiones) for c in ("train", "val", "test")}}
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2), flush=True)
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen", type=Path, required=True)
    parser.add_argument("--destino", type=Path, default=RAIZ)
    argumentos = parser.parse_args()
    preparar(argumentos.origen, argumentos.destino)
