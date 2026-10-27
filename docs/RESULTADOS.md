# Resultados del entrenamiento de señales

Corrida `senales_1`: cinco CNN con base ImageNet congelada y cabeza de cuatro clases, máximo 10 épocas. Semilla 42. Ejecución en CPU con TensorFlow 2.21.0 y Python 3.12.10.

538 fotos originales; 20 HEIC convertidas; 3 copias duplicadas y 2 fotos de zona escolar sin propuesta válida excluidas. Resultado: **533 recortes**, divididos en **372 entrenamiento, 107 validación y 54 prueba**.

| Modelo | Acierto prueba | F1 macro prueba | F1 macro validación | Latencia mediana | Cobertura confianza ≥0,80 |
|---|---:|---:|---:|---:|---:|
| VGG16 | 85,19 % | 0,8484 | 0,9629 | 76,4 ms | 88,89 % |
| VGG19 | 87,04 % | 0,8713 | 0,9431 | 120,3 ms | 83,33 % |
| MobileNet | 88,89 % | 0,8894 | 0,9622 | 74,9 ms | 85,19 % |
| ResNet50 | 96,30 % | 0,9629 | 0,9720 | 129,9 ms | 83,33 % |
| DenseNet121 | 90,74 % | 0,9082 | 0,9620 | 407,0 ms | 72,22 % |

**Selección por validación: ResNet50** (F1 macro; desempate por menor pérdida). **Modelo predeterminado del robot: MobileNet**, según el plan. Se puede elegir ResNet50 con `--modelo resultados/senales_1/ResNet50/ResNet50.keras`.

Las latencias corresponden a una inferencia completa, una imagen y diez repeticiones después de calentamiento, con ocho hilos intraoperación. Excluyen lectura, resize, búsqueda de candidatos, captura y transmisión; no representan los FPS del sistema. Hasta cuatro candidatos implican varias inferencias por fotograma.

La cobertura indica la fracción de fotos con confianza ≥0,80, independientemente de si la clasificación fue correcta. El acierto de la tabla corresponde a todas las fotos de prueba y no debe confundirse con la precisión de las detecciones confirmadas del robot.

Se recargaron y evaluaron los cinco archivos `.keras`; sus resultados coinciden con la evaluación del entrenamiento. Matrices, curvas, probabilidades individuales y pesos completos permanecen en `resultados/senales_1/`, excluidos de Git. La tabla numérica está en [resultados.csv](resultados.csv).

## Validación sin hardware

- **73 pruebas pasan**, incluyendo las originales y las pruebas de modelos exportados, etiquetas, RGB/BGR, intervalos, prioridad de parada, confirmación, rearme y descarte de pendientes.
- La demostración `python -m pruebas.demostracion_cnn` usa fotos de validación reales en escenas compuestas, con línea y reloj simulados. Registra intervalos de 1/0,5/2 segundos y PARE desde el segundo 14 hasta el 17. Artefactos en `resultados/demostracion/`.
- La prueba `python -m pruebas.probar_pipeline_cnn` ejecuta `main.py` con un video compuesto de **400 fotogramas** a ritmo real, salida de registro y ROI ajustadas temporalmente para esas escenas. Reconoce las cuatro señales, emite avances con los tres intervalos y realiza PARE durante tres segundos. También ejercita la pérdida de línea al llegar a la escena vacía. No modifica la calibración del robot.

## Límites de la evaluación

- Solo 54 fotos en prueba, tomadas de la misma pista y de sesiones similares. No demuestra generalización a otras pistas, fondos, cámaras o iluminación.
- Se eliminan duplicados exactos, pero pueden persistir fotos cercanas de una secuencia entre particiones. Una evaluación independiente debería separar sesiones y utilizar videos nuevos.
- La localización es por color y geometría; las métricas del clasificador no incluyen las dos fotos rechazadas por falta de recorte.
- No hay clase de fondo u objeto desconocido. El umbral y la confirmación temporal reducen reacciones espurias, pero no garantizan rechazar objetos ajenos.
- No se conectaron cámara ni robot físico. Los intervalos necesitan calibración en el montaje real.

Los originales y resultados del proyecto padre se conservaron. Fotos, recortes y pesos no se publican en Git.
