# Reglas de esta extensión CNN

Leer `README.md`: es la documentación vigente. El usuario autorizó expresamente usar redes neuronales y el entrenamiento del proyecto `5_cnns`; las restricciones del reto 1 quedan como antecedente en `docs/CLAUDE_RETO1.md` y no prohíben este trabajo.

- Mantener Python 3.12, nombres/comentarios en español y módulos sencillos.
- Parámetros ajustables del robot en `config.py`.
- Mantener separadas percepción, decisiones, entrenamiento y comunicación.
- El orden de clases es exclusivamente 0=PARE, 1=SIGA, 2=VELOCIDAD_60, 3=ZONA_ESCOLAR.
- No modificar las fotos ni los resultados del proyecto padre.
- No incluir fotos ni pesos en Git; guardar su procedencia y particiones.
- No ajustar hiperparámetros con la partición de prueba.
- Verificar cambios de control con pruebas sin hardware antes de conectar el robot.
- Documentar los parámetros que requieren calibración física y no presentar simulaciones como pruebas con el robot.
