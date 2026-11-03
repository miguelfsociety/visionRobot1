"""Reentrena ResNet50 con giros e iluminación, conservando las particiones.

Las variantes se crean únicamente para entrenamiento. El video del debug
no se incorpora al dataset y la selección se decide solo con validación.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from entrenamiento.main import EntrenadorSenales, RAIZ
from vision.imagenes_cnn import leer_rgb, preparar_lote

VARIANTES = ((0, 1.0), (1, 0.8), (2, 1.0), (3, 1.15))


class EntrenadorConVariaciones(EntrenadorSenales):
    """Extiende el entrenador existente sin cambiar los datos de validación/prueba."""

    def _cargar_datos(self):
        super()._cargar_datos()
        self.firma_original = self.firma
        self.firma = hashlib.sha256((self.firma + repr(VARIANTES)).encode()).hexdigest()
        self.grupos["train"] = [dict(fila, giro=giro, brillo=brillo)
                                for fila in self.grupos["train"] for giro, brillo in VARIANTES]
        with (Path(self.exp_dir) / "variantes_train.csv").open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(self.grupos["train"][0]))
            escritor.writeheader()
            escritor.writerows(self.grupos["train"])

    def _datos(self, filas, preprocesar):
        """Aplica variaciones deterministas solo a filas de entrenamiento."""
        def generar():
            for fila in filas:
                imagen = leer_rgb(RAIZ / fila["ruta"])
                imagen = np.clip(np.rot90(imagen, fila.get("giro", 0)) * fila.get("brillo", 1.0), 0, 255)
                yield np.asarray(preparar_lote(imagen, self.img_size, preprocesar))[0]
        datos = tf.data.Dataset.from_generator(generar, output_signature=tf.TensorSpec((*self.img_size, 3), tf.float32))
        datos = datos.apply(tf.data.experimental.assert_cardinality(len(filas)))
        opciones = tf.data.Options()
        opciones.threading.private_threadpool_size = 2
        return datos.batch(self.batch_size).with_options(opciones).prefetch(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", default="senales_iteracion_2")
    parser.add_argument("--epocas", type=int, default=20)
    args = parser.parse_args()
    if (RAIZ / "resultados" / args.experimento).resolve() == (RAIZ / "resultados/senales_1").resolve():
        raise ValueError("Usa otro experimento para conservar los modelos iniciales")
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    entrenador = EntrenadorConVariaciones(dataset_dir=str(RAIZ / "datos_preparados"),
        output_dir=str(RAIZ / "resultados"), experimento=args.experimento,
        max_epochs=args.epocas, patience=5, batch_size=32)
    exp = Path(entrenador.exp_dir)
    config = {"arquitectura": "ResNet50", "max_epochs": args.epocas, "patience": 5,
              "learning_rate": 1e-3, "batch_size": 32, "img_size": [224, 224],
              "semilla": 42, "base_congelada": True, "variantes": VARIANTES,
              "particiones_originales_sha256": entrenador.firma_original,
              "muestras": {k: len(v) for k, v in entrenador.grupos.items()},
              "video_usado_para_entrenar": False}
    (exp / "configuracion.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    resultado = entrenador.entrenar("ResNet50")
    metadatos_path = exp / "ResNet50/metadatos.json"
    metadatos = json.loads(metadatos_path.read_text(encoding="utf-8"))
    metadatos["particiones_sha256"] = entrenador.firma_original
    metadatos["variantes_train"] = VARIANTES
    metadatos_path.write_text(json.dumps(metadatos, indent=2), encoding="utf-8")
    entrenador._guardar_csv([resultado], exp / "resultados.csv")
    anterior_path = RAIZ / "resultados/senales_1/ResNet50/metricas.json"
    anterior = json.loads(anterior_path.read_text(encoding="utf-8"))
    # Misma regla que en la primera corrida; nunca seleccionar con test.
    mejora = (resultado["val_f1_macro"], -resultado["val_loss"]) > (anterior["val_f1_macro"], -anterior["val_loss"])
    seleccion = {"criterio": "F1 macro de validación; desempate por menor val_loss",
                 "val_f1_anterior": anterior["val_f1_macro"], "val_f1_nuevo": resultado["val_f1_macro"],
                 "usar_reentrenado": mejora,
                 "modelo_recomendado": str((exp / "ResNet50/ResNet50.keras" if mejora else anterior_path.with_name("ResNet50.keras")).relative_to(RAIZ))}
    (exp / "seleccion.json").write_text(json.dumps(seleccion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(seleccion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
