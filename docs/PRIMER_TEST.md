# Revisión de primer_test.mp4 y segunda ronda

## Evidencia de la primera ronda

Se revisó la grabación de pantalla `primer_test.mp4`: 120,1 segundos, 30 FPS y resolución 1060×1394. La imagen del debug muestra 1 segundo entre avances inicialmente, 0,5 después de reconocer 60 y 2 segundos después de zona escolar. El intervalo escolar se conserva durante los tramos siguientes hasta otra señal de velocidad, como estaba especificado.

La telemetría disponible tiene 13.391 fotogramas durante 177,516 segundos: FPS mediano 62,5; latencia mediana del ciclo 4,2 ms y p95 de 12,4 ms. El registro es más largo que la grabación, por lo que sus tiempos no se alinean directamente. La etapa `senales` anterior solo medía la entrega/consulta del trabajador asíncrono; no medía el tiempo real de la CNN. Por tanto, los 4,2 ms no son latencia completa de percepción ni prueban la velocidad del stream.

El diagnóstico principal de lentitud es la separación entre pulsos, coherente con lo observado en el video. Un pulso de avance del firmware dura aproximadamente 100 ms; esperar 1 o 2 segundos entre avances deja largos periodos detenido en rectas. No se midieron cm/s ni tiempos del firmware en esta revisión.

Se observa una falsa detección de zona escolar sobre un triángulo de madera cerca del segundo 90. En ese fotograma no está en distancia de disparo. También aparecen propuestas sobre objetos del entorno: el clasificador sigue teniendo cuatro clases y no garantiza rechazar fondos desconocidos.

Los registros originales y la configuración se copiaron antes de las pruebas a `resultados/revision_primer_test/`. La grabación permanece intacta y excluida de Git.

## Cambios para la siguiente ronda

El usuario eligió estos intervalos:

| Perfil | Inicial/SIGA | 60 | Zona escolar |
|---|---:|---:|---:|
| `original` | 1 s | 0,5 s | 2 s |
| `agil` | 0,25 s | 0,125 s | 0,5 s |

`agil` es el perfil predeterminado de `main.py`. En rectas solicita cuatro veces más pulsos que el original; esto no equivale a demostrar cuatro veces más velocidad física. La frecuencia global sigue limitada a 8 Hz y PARE conserva sus tres segundos y el intervalo anterior al reanudar.

- La alternancia de giro/avance se confirma con los pulsos admitidos, evitando que una cámara de más FPS consuma el plan sin emitirlo. Mientras espera el intervalo de avance, sigue corrigiendo giros si la línea lo requiere.
- SIGA se registra una vez por aparición; antes se alternaban estados repetidamente mientras permanecía visible.
- Al pulsar `i` se reinician la máquina y el programador para que una señal PARE vista durante la preparación no consuma sus tres segundos antes de iniciar.
- Las propuestas exigen al menos cuatro vértices aproximados. Se conserva la admisión de señales circulares y pentagonales, y se descartan triángulos de madera como el observado.
- La CNN se traza y calienta una vez con `tf.function`. En una comparación en el mismo proceso y equipo, el modelo reentrenado pasó de 367,5 ms en ejecución directa a 69,2 ms en grafo, mediana de 20 repeticiones, excluyendo lectura/resize. La arquitectura no se hizo más pequeña y el resultado no equivale a FPS del sistema completo.
- Cada ejecución conserva sus registros en una carpeta nueva e incluye configuración, decisiones, tiempo del trabajador CNN, edad de resultados y comandos despachados por Bluetooth. Ya no se sobrescriben las rondas anteriores.

## Reentrenamiento

Se reentrenó ResNet50 usando los mismos 372 originales de entrenamiento, con cuatro variantes por foto: giro 0°/brillo 1; 90°/0,8; 180°/1; 270°/1,15. Son 1.488 muestras derivadas exclusivamente del entrenamiento. Las 107 fotos de validación y 54 de prueba permanecen iguales.

La base ImageNet sigue congelada. Se entrenó una cabeza nueva, máximo 20 épocas, paciencia 5, semilla 42; terminó en la época 11 y restauró la época 6 por menor pérdida de validación. No se entrenó con fotogramas del debug ni se usaron predicciones automáticas como etiquetas.

| Métrica | ResNet50 inicial | Reentrenado |
|---|---:|---:|
| F1 macro de validación | 0,9720 | 0,9815 |
| Pérdida de validación | 0,1081 | 0,0883 |
| Acierto de prueba | 96,30 % | 96,30 % |
| Recall de PARE en prueba | 100 % | 100 % |
| Precisión de PARE en prueba | 92,86 % | 86,67 % |

Se selecciona el nuevo modelo por validación. La mejora no es uniforme: ambos fallan dos de 54 fotos de prueba, y el nuevo tiene más falsos PARE en ese conjunto. Hay que observar paradas indebidas en la siguiente ronda. No se garantiza una mejora del recorrido físico a partir de esta evaluación.

Archivos nuevos: `resultados/senales_iteracion_2/ResNet50/ResNet50.keras` y `metadatos.json`. Los pesos iniciales siguen disponibles en `resultados/senales_1/ResNet50/`.

## Reproducción y siguiente prueba

Validación local: **84 pruebas aprobadas** y ejecución del flujo completo con el modelo reentrenado sobre **400 fotogramas compuestos**, con 108 comandos registrados y sin hardware. El modelo exportado se recargó y conservó el 96,30 % de acierto de prueba. Estas verificaciones no sustituyen la segunda ronda física.

Desde `visionRobot1`, con `MAC_MBOT` ya configurada en `.env`:

```powershell
..\.venv\Scripts\python.exe main.py --url "http://192.168.137.190:4747/video" --modelo "resultados/senales_iteracion_2/ResNet50/ResNet50.keras" --perfil agil --etapa completo --salida mbot --debug
```

Presionar `i` para iniciar y `q`/Esc para terminar. Probar primero recta/curva y luego las cuatro señales. La combinación de mayor cadencia y correcciones nuevas todavía requiere validación física; revisar si aparece desenfoque, salida de línea o reacción tardía.

Para comparar, repetir el mismo tramo con `--perfil original`. Mantener cámara, iluminación y posición inicial. Entregar el nuevo video y la carpeta de registros impresa al iniciar.

```powershell
# Reentrenar sin sobrescribir el primer experimento:
..\.venv\Scripts\python.exe -m entrenamiento.reentrenar
# Pruebas de regresión y flujo completo sin hardware:
..\.venv\Scripts\python.exe -m unittest discover -s pruebas -v
..\.venv\Scripts\python.exe -m pruebas.probar_pipeline_cnn --modelo resultados/senales_iteracion_2/ResNet50/ResNet50.keras --perfil agil
```

La auditoría del video muestrea una imagen por segundo, extrayendo solo la zona de cámara `(18,57,738,1337)`. Sirve para revisar candidatos y orientación, pero los dibujos superpuestos y la compresión alteran los recortes. No se presenta una exactitud del video sin etiquetas humanas ni se incorporan esas imágenes al entrenamiento.
