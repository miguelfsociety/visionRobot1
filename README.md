# Robot seguidor de línea con cinco CNN de señales

Extensión de [visionRobot1, rama feature/control-movimiento](https://github.com/stfrancoc/visionRobot1/tree/feature/control-movimiento), basada en el commit `77a695982a9b997e79e8e5823f89c7a39527fac1`. Integra el entrenador del proyecto `5_cnns` con el seguimiento de línea, la máquina de estados y la comunicación Bluetooth del robot original.

## Señales y comportamiento

| Carpeta | Etiqueta de salida | Acción |
|---|---|---|
| `dataset/0` | `PARE` | Parar 3 segundos; reanudar con el intervalo anterior |
| `dataset/1` | `SIGA` | Un pulso de avance cada 1 segundo |
| `dataset/2` | `VELOCIDAD_60` | Un pulso de avance cada 0,5 segundos |
| `dataset/3` | `ZONA_ESCOLAR` | Un pulso de avance cada 2 segundos |

Los intervalos controlan la frecuencia de `Robot.adelante()`. El firmware recibe pulsos discretos; la etiqueta de 60 km/h no implica esa velocidad física. Los giros mantienen su propia frecuencia de corrección. El último intervalo persiste si no hay señal; el inicial es 1 segundo.

## Instalación y datos

Python 3.12. Desde esta carpeta puede reutilizarse el entorno del proyecto padre:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m entrenamiento.preparar --origen ..\dataset
```

En una máquina nueva, crear un entorno e instalar los requisitos antes de ejecutar los mismos módulos con su Python.

`entrenamiento.preparar` lee solo las carpetas `0–3`, corrige EXIF, convierte HEIC, copia JPEG a `dataset/` y genera recortes en `datos_preparados/`. Conserva las fotos de origen. El detector propone regiones rojas, verdes y amarillas, con filtros de área/aspecto/extensión; la CNN determina la clase. Se usa el mismo localizador en fotografías y video. Una foto sin propuesta válida se registra y excluye; no se sustituye silenciosamente por el fondo completo.

Artefactos de preparación:

- `manifiesto.json`: procedencia, hashes, conversiones, duplicados y rectángulos.
- `rechazados.json`: errores y fotos sin recorte válido.
- `particiones.csv`: división estratificada persistente 70/20/10, semilla 42.
- `resumen.json`: conteos y etiquetas.

Los duplicados exactos por archivo o píxeles no se repiten entre particiones. La división sucede antes de cualquier eventual aumento de datos. Esta versión no usa aumento: entrena la cabeza sobre características congeladas. La lista del manifiesto es la fuente de verdad; archivos antiguos no referenciados no participan en el entrenamiento.

## Entrenar las cinco CNN

```powershell
..\.venv\Scripts\python.exe -m entrenamiento.main
# Continuar una corrida interrumpida sin repetir modelos completados:
..\.venv\Scripts\python.exe -m entrenamiento.main --reanudar
# Ejemplo de corrida distinta:
..\.venv\Scripts\python.exe -m entrenamiento.main --modelos MobileNet --experimento prueba_mobile --epocas 10
```

Se entrenan VGG16, VGG19, MobileNet, ResNet50 y DenseNet121, secuencialmente. Configuración inicial: entrada 224×224 RGB, lote 32, Adam `1e-3`, Dropout 0,3, máximo 10 épocas y paciencia 5 por `val_loss`, restaurando los mejores pesos. ImageNet se descarga solo si no está en la caché de Keras.

La base ImageNet está congelada. Sus características con global average pooling se calculan una vez y se guardan en disco; la cabeza Dropout + Dense aprende sobre ellas. Es equivalente a repetir esa misma base congelada en cada época y reduce el costo de CPU. No se cachean todas las fotos originales en RAM. Para cambiar la arquitectura, el preprocesamiento o el tamaño, usar un experimento nuevo y regenerar las características.

`resultados/senales_1/` contiene la configuración, copia de las particiones, comparación CSV, curvas, selección por F1 macro de **validación**, y una carpeta por modelo con:

- `<modelo>.keras` y `metadatos.json`: modelo completo con pesos y contrato de etiquetas/preprocesamiento.
- Historial, métricas, reporte por clase, matriz de confusión y curvas de pérdida.
- `predicciones_test.csv`: las probabilidades de cada foto de prueba.
- Latencia mediana y p95 sobre diez inferencias completas, después de calentamiento; excluye lectura y resize y no representa los FPS completos del robot.

TensorFlow instalado en Windows usa CPU. Los resultados provienen de fotos similares de una misma pista/sesión; deben verificarse con nuevos videos y otras condiciones de iluminación antes de atribuirles capacidad de generalización.

## Evaluar y predecir sin entrenar

```powershell
..\.venv\Scripts\python.exe -m entrenamiento.evaluar
..\.venv\Scripts\python.exe -m entrenamiento.evaluar --modelo resultados/senales_1/ResNet50/ResNet50.keras
..\.venv\Scripts\python.exe predecir.py "..\dataset\0\20261027_095759.jpg"
..\.venv\Scripts\python.exe predecir.py "foto.jpg" --modelo resultados/senales_1/ResNet50/ResNet50.keras
..\.venv\Scripts\python.exe predecir.py "recorte.jpg" --recorte
```

La evaluación recarga el archivo exportado y escribe `evaluacion_recargada.json`; incluye la fracción de fotos con confianza ≥0,80. El CLI de predicción devuelve una lista de candidatos con sus probabilidades; una lista vacía significa que no encontró una región de señal. No aplica confirmación temporal a una foto aislada.

Uso de la clase Python:

```python
from vision.clasificador_senales import ClasificadorSenales

predictor = ClasificadorSenales("resultados/senales_1/MobileNet/MobileNet.keras")
resultado = predictor.predecir("recorte.jpg")
# O un recorte de OpenCV:
# resultado = predictor.predecir(recorte_bgr, orden_color="BGR")
print(resultado["identificador"], resultado["nombre"], resultado["confianza"])
```

La clase clasifica una imagen de señal ya recortada y carga los pesos una sola vez. Para fotografías completas, usar primero `vision.propuestas_senales.proponer_senales`, como hace el CLI. Matrices RGB/BGR deben tener tres canales y valores en [0,255]. Mantener `metadatos.json` junto al `.keras`; inferencia no necesita dataset, Internet ni pesos ImageNet externos. El softmax siempre elige entre cuatro clases; la confianza por sí sola no garantiza rechazar objetos desconocidos.

## Ejecutar el robot

```powershell
# Solo percepción, sin mover el robot:
..\.venv\Scripts\python.exe main.py --camara 0 --debug
# Mostrar decisiones, sin enviar:
..\.venv\Scripts\python.exe main.py --video videos/ensayo.mp4 --etapa comandos --salida nula --debug
# Registrar pulsos, sin Bluetooth:
..\.venv\Scripts\python.exe main.py --url http://192.168.1.50:8080/video --etapa completo --salida registro --debug
# Robot físico: presionar 'i' en la ventana para iniciar:
..\.venv\Scripts\python.exe main.py --url http://192.168.1.50:8080/video --etapa completo --salida mbot --debug
```

`--modelo` permite seleccionar cualquiera de las cinco CNN. MobileNet es la predeterminada para el robot. `--detector clasico` conserva el detector PARE/SIGA del reto anterior para comparación y calibración.

La CNN corre en un trabajador con una sola casilla: un fotograma nuevo reemplaza al pendiente; nunca se acumula una cola. La confirmación exige confianza ≥0,80 y presencia en 3 de las últimas 5 **inferencias nuevas**, además de cruzar `Y_DISPARO`. Un resultado caduca al superar un segundo desde la entrega del fotograma al trabajador. Volver a consultar el mismo resultado no cuenta como otra inferencia. PARE solo se rearma después de cinco inferencias consecutivas sin PARE visible. La parada tiene prioridad sobre los pulsos y descarta movimientos pendientes.

`--salida registro` escribe `resultados/comandos.csv`, incluyendo señal, confianza, estado e intervalo. La telemetría del ciclo se escribe en `resultados/telemetria/`. Los modos `vision` y `comandos` no abren Bluetooth ni envían comandos. La falta de un modelo válido produce un error; no cambia automáticamente al detector clásico.

Calibrar con el robot: `ROTACION`, `FILA_CHASIS`, `ROI_SENALES_*`, `Y_DISPARO`, rangos `CNN_RANGOS_HSV`, confianza, caducidad y `INTERVALOS_AVANCE`. Los valores de orientación/ROI se heredaron del robot original. La zona recortada debe permitir ver una señal completa antes de que salga de la ROI; la configuración depende del montaje de la cámara. Los giros del firmware también desplazan el robot y afectan la velocidad efectiva.

## Pruebas y entrega

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s pruebas -v
# Fotos reales de validación, reloj y línea simulados, sin Bluetooth:
..\.venv\Scripts\python.exe -m pruebas.demostracion_cnn
# Flujo main.py con 400 fotogramas compuestos y reloj real (~20 segundos):
..\.venv\Scripts\python.exe -m pruebas.probar_pipeline_cnn
```

Incluye las pruebas originales y pruebas de intervalos, prioridad de parada, PARE de tres segundos, rearme, confirmación, ausencia/baja confianza, descarte de fotogramas, particiones y exportaciones de los cinco modelos. Las pruebas de pesos reales se omiten si todavía no se entrenaron. No se ha validado con cámara ni robot físico.

Fotos, recortes, cachés y pesos quedan disponibles localmente y excluidos de Git. El código y los informes resumidos de `docs/` permiten revisar el trabajo; para reproducirlo, proporcionar las fotos y ejecutar preparación/entrenamiento. La documentación del reto anterior queda como antecedente en `docs/ROBOT_ORIGINAL.md`.

Para completar el fork remoto cuando la cuenta esté autenticada:

```powershell
gh auth login --hostname github.com --web
gh auth switch --user miguelfsociety
gh repo fork stfrancoc/visionRobot1 --clone=false
git remote add origin https://github.com/miguelfsociety/visionRobot1.git
git push -u origin feature/cnn-senales
```

La copia local conserva `upstream` apuntando al repositorio original. El fork remoto y su publicación requieren credenciales válidas de `miguelfsociety`.
