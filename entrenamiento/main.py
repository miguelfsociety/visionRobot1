"""Entrena y evalúa las cinco CNN usando una partición persistente."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .base import CNNTrainer, MODELOS
import keras
from keras import layers
import numpy as np
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf

from vision.catalogo import CLASES
from vision.imagenes_cnn import preparar_lote

RAIZ = Path(__file__).resolve().parents[1]


class EntrenadorSenales(CNNTrainer):
    """Reutiliza informes/gráficas del proyecto original y sus arquitecturas.

La base congelada se evalúa una vez por imagen y sus características se
guardan en disco. Entrenar la cabeza sobre ellas equivale a ejecutar la
base congelada en cada época, sin aumento de datos ni recalcularla.
"""

    def _cargar_datos(self):
        ruta = RAIZ / "datos_preparados" / "particiones.csv"
        if not ruta.is_file():
            raise ValueError("Primero ejecuta python -m entrenamiento.preparar --origen ../dataset")
        with ruta.open(encoding="utf-8") as archivo:
            self.filas = list(csv.DictReader(archivo))
        self.class_names = list(CLASES)
        self.num_classes = len(CLASES)
        self.firma = hashlib.sha256(ruta.read_bytes()).hexdigest()
        self.grupos = {c: [r for r in self.filas if r["conjunto"] == c] for c in ("train", "val", "test")}
        if any(not grupo for grupo in self.grupos.values()):
            raise ValueError("Las tres particiones deben tener muestras")
        vistas = set()
        for fila in self.filas:
            if fila["sha256"] in vistas:
                raise ValueError("Hay imágenes repetidas en las particiones")
            vistas.add(fila["sha256"])
            if not 0 <= int(fila["clase"]) < 4:
                raise ValueError("Solo se admiten clases 0, 1, 2 y 3")
        shutil.copy2(ruta, Path(self.exp_dir) / "particiones.csv")

    def _datos(self, filas, preprocesar):
        """Carga de forma perezosa, sin cachear imágenes de alta resolución."""
        def generar():
            for fila in filas:
                yield np.asarray(preparar_lote(RAIZ / fila["ruta"], self.img_size, preprocesar))[0]
        datos = tf.data.Dataset.from_generator(generar, output_signature=tf.TensorSpec((*self.img_size, 3), tf.float32))
        datos = datos.apply(tf.data.experimental.assert_cardinality(len(filas)))
        opciones = tf.data.Options()
        opciones.threading.private_threadpool_size = 2
        return datos.batch(self.batch_size).with_options(opciones).prefetch(1)

    def entrenar(self, nombre):
        """Extrae características, entrena la cabeza y evalúa prueba una vez."""
        keras.utils.set_random_seed(self.seed)
        constructor, preprocesar = MODELOS[nombre]
        base = constructor(weights="imagenet", include_top=False, input_shape=(*self.img_size, 3))
        base.trainable = False
        extractor = keras.Model(base.input, layers.GlobalAveragePooling2D()(base.output))
        carpeta = Path(self.exp_dir) / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        cache = Path(self.exp_dir) / "caracteristicas" / nombre
        cache.mkdir(parents=True, exist_ok=True)
        caracteristicas, etiquetas = {}, {}
        for conjunto, filas in self.grupos.items():
            archivo_cache = cache / f"{conjunto}_{self.firma[:16]}.npy"
            if archivo_cache.is_file():
                caracteristicas[conjunto] = np.load(archivo_cache)
            else:
                print(f"{nombre}: extrayendo {conjunto} ({len(filas)} imágenes)", flush=True)
                caracteristicas[conjunto] = extractor.predict(self._datos(filas, preprocesar), verbose=1)
                np.save(archivo_cache, caracteristicas[conjunto])
            etiquetas[conjunto] = np.array([int(f["clase"]) for f in filas], dtype=np.int32)
        entrada = keras.Input(shape=(extractor.output_shape[-1],))
        salida = layers.Dense(4, activation="softmax")(layers.Dropout(0.3)(entrada))
        cabeza = keras.Model(entrada, salida, name="cabeza_senales")
        cabeza.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        def ds(conjunto, barajar=False):
            datos = tf.data.Dataset.from_tensor_slices((caracteristicas[conjunto], etiquetas[conjunto]))
            if barajar:
                datos = datos.shuffle(len(etiquetas[conjunto]), seed=self.seed)
            opciones = tf.data.Options()
            opciones.threading.private_threadpool_size = 2
            return datos.batch(self.batch_size).with_options(opciones).prefetch(1)
        historial = cabeza.fit(ds("train", True), validation_data=ds("val"), epochs=self.max_epochs,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)],
            shuffle=False, verbose=2)
        mejor = int(np.argmin(historial.history["val_loss"])) + 1
        self.historiales[nombre] = historial.history
        probabilidades = np.asarray(cabeza(caracteristicas["test"], training=False))
        predichas = probabilidades.argmax(axis=1)
        reales = etiquetas["test"]
        val_pred = np.asarray(cabeza(caracteristicas["val"], training=False)).argmax(axis=1)
        grafica_perdida = self._graficar_perdida(nombre, historial.history, mejor, str(carpeta))
        grafica_cm = self._graficar_matriz_confusion(nombre, reales, predichas, str(carpeta))
        _, reporte = self._guardar_classification_report(nombre, reales, predichas, str(carpeta), len(historial.history["loss"]), mejor)
        modelo = keras.Model(extractor.input, cabeza(extractor.output), name=nombre)
        modelo.save(carpeta / f"{nombre}.keras")
        metadatos = {"arquitectura": nombre, "clases": list(CLASES), "tamano": list(self.img_size),
                     "preprocesamiento": nombre, "orden_color": "RGB", "particiones_sha256": self.firma}
        (carpeta / "metadatos.json").write_text(json.dumps(metadatos, indent=2), encoding="utf-8")
        (carpeta / "historial.json").write_text(json.dumps(historial.history, indent=2), encoding="utf-8")
        (carpeta / "reporte_clasificacion.json").write_text(json.dumps(reporte, indent=2), encoding="utf-8")
        with (carpeta / "predicciones_test.csv").open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(["ruta", "real", "prediccion", *CLASES])
            for fila, real, predicha, probs in zip(self.grupos["test"], reales, predichas, probabilidades):
                escritor.writerow([fila["ruta"], int(real), int(predicha), *map(float, probs)])
        lote = preparar_lote(RAIZ / self.grupos["test"][0]["ruta"], self.img_size, preprocesar)
        # Incluye la ejecución completa de la base; excluye lectura y resize.
        modelo(lote, training=False).numpy()
        latencias = []
        for _ in range(10):
            inicio = time.perf_counter()
            modelo(lote, training=False).numpy()
            latencias.append((time.perf_counter() - inicio) * 1000)
        resultado = {"modelo": nombre, "experimento": self.experimento, "max_epochs": self.max_epochs,
            "epoca_parada": len(historial.history["loss"]), "mejor_epoca": mejor,
            "train_loss": historial.history["loss"][mejor-1], "val_loss": historial.history["val_loss"][mejor-1],
            "val_f1_macro": f1_score(etiquetas["val"], val_pred, average="macro", zero_division=0),
            "accuracy": float(np.mean(reales == predichas)),
            "f1_macro": f1_score(reales, predichas, labels=range(4), average="macro", zero_division=0),
            "f1_ponderado": f1_score(reales, predichas, labels=range(4), average="weighted", zero_division=0),
            "precision_macro": reporte["macro avg"]["precision"], "recall_macro": reporte["macro avg"]["recall"],
            "latencia_mediana_ms": float(np.median(latencias)), "latencia_p95_ms": float(np.percentile(latencias, 95)),
            "grafica_perdida": grafica_perdida, "grafica_confusion": grafica_cm}
        (carpeta / "metricas.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
        return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--experimento", default="senales_1")
    parser.add_argument("--hilos", type=int, default=8)
    parser.add_argument("--reanudar", action="store_true", help="Reutiliza modelos ya terminados con la misma configuración")
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.hilos)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    entrenador = EntrenadorSenales(dataset_dir=str(RAIZ / "datos_preparados"), output_dir=str(RAIZ / "resultados"),
        experimento=args.experimento, max_epochs=args.epocas, batch_size=args.lote, patience=5)
    configuracion = {**vars(args), "semilla": 42, "learning_rate": 1e-3, "base_congelada": True,
        "img_size": [224, 224], "patience": 5, "dropout": 0.3, "val_split": 0.2, "test_split": 0.1,
        "particiones_sha256": entrenador.firma, "clases": list(CLASES), "tensorflow": tf.__version__,
        "dispositivos": [d.name for d in tf.config.list_physical_devices()],
        "limitacion": "Fotos correlacionadas de una misma pista/sesión; no prueban generalización a otras pistas."}
    exp = Path(entrenador.exp_dir)
    config_path = exp / "configuracion.json"
    if args.reanudar and config_path.exists():
        anterior = json.loads(config_path.read_text(encoding="utf-8"))
        for clave in ("particiones_sha256", "epocas", "lote", "semilla"):
            if anterior[clave] != configuracion[clave]:
                raise ValueError(f"No se puede reanudar: cambió {clave}")
    config_path.write_text(json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8")
    resultados = []
    for nombre in args.modelos:
        metricas = exp / nombre / "metricas.json"
        if args.reanudar and metricas.exists() and (exp / nombre / f"{nombre}.keras").exists():
            resultado = json.loads(metricas.read_text(encoding="utf-8"))
        else:
            print(f"\nEntrenando {nombre}", flush=True)
            resultado = entrenador.entrenar(nombre)
        resultados.append(resultado)
        entrenador._guardar_csv(resultados, exp / "resultados.csv")
        keras.backend.clear_session()
    if len(entrenador.historiales) > 1:
        entrenador._graficar_comparativa(entrenador.historiales)
    mejor = max(resultados, key=lambda r: (r["val_f1_macro"], -r["val_loss"]))
    (exp / "seleccion.json").write_text(json.dumps({"mejor_validacion": mejor["modelo"],
        "criterio": "F1 macro de validación; desempate por menor val_loss", "robot_predeterminado": "MobileNet"}, indent=2), encoding="utf-8")
    print(f"Comparación terminada: {exp / 'resultados.csv'}", flush=True)


if __name__ == "__main__":
    main()
