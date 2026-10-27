"""Detección CNN en trabajador con casilla de último fotograma.

Complejidad por inferencia: O(alto × ancho) más hasta CNN_MAX_CANDIDATOS
ejecuciones de la CNN. No hay una cola creciente de fotogramas.
"""

from collections import deque
import threading
import time

import config
from control.contratos import ResultadoSenales
from vision.propuestas_senales import proponer_senales


class DetectorSenalesCNN:
    """Clasifica candidatos y confirma por inferencias nuevas, nunca por FPS."""

    def __init__(self, clasificador, cfg=config):
        self.clasificador = clasificador
        self.cfg = cfg
        self.historial = deque(maxlen=cfg.CNN_CONFIRMAR_M)
        self.secuencia = 0

    def detectar(self, roi_bgr, desplazamiento_y=0):
        """Recibe ROI BGR y devuelve detecciones en coordenadas del fotograma."""
        candidatos, validos = [], []
        for propuesta in proponer_senales(roi_bgr, self.cfg):
            resultado = self.clasificador.predecir(propuesta.pop("recorte"), "BGR")
            x, y, w, h = propuesta["bounding_rect"]
            cx, cy = propuesta["centroide"]
            propuesta.update(color=resultado["nombre"], confianza=resultado["confianza"],
                bounding_rect=(x, y + desplazamiento_y, w, h), centroide=(cx, cy + desplazamiento_y))
            candidatos.append(propuesta)
            if resultado["confianza"] >= self.cfg.CNN_CONFIANZA_MINIMA:
                validos.append(propuesta)
        nombres = {c["color"] for c in validos}
        self.historial.append(nombres)
        self.secuencia += 1
        confirmados = [c for c in validos if sum(c["color"] in h for h in self.historial) >= self.cfg.CNN_CONFIRMAR_N]
        # PARE tiene prioridad entre señales que ya están en distancia de actuación.
        disparo = desplazamiento_y + roi_bgr.shape[0] * self.cfg.Y_DISPARO
        cercanos = [c for c in confirmados if c["centroide"][1] >= disparo]
        elegido = max(cercanos or confirmados, key=lambda c: (c["color"] == "PARE", c["area"]), default=None)
        return ResultadoSenales(senal=elegido["color"] if elegido else None,
            en_disparo=bool(elegido and elegido["centroide"][1] >= disparo),
            candidatos=candidatos, confianza=elegido["confianza"] if elegido else 0.0,
            actualizacion=self.secuencia, pare_visible="PARE" in nombres)


class TrabajadorSenales:
    """Procesa solo el último fotograma disponible, sin bloquear la visión."""

    def __init__(self, detector, cfg=config):
        self.detector, self.cfg = detector, cfg
        self._condicion = threading.Condition()
        self._pendiente = None
        self._resultado = None
        self._error = None
        self._cerrado = False
        self._hilo = threading.Thread(target=self._procesar, daemon=True)
        self._hilo.start()

    def enviar(self, roi_bgr, desplazamiento_y=0):
        """Reemplaza la casilla pendiente con una copia del último fotograma."""
        with self._condicion:
            if not self._cerrado:
                self._pendiente = (roi_bgr.copy(), desplazamiento_y, time.monotonic())
                self._condicion.notify()

    def _procesar(self):
        try:
            while True:
                with self._condicion:
                    self._condicion.wait_for(lambda: self._cerrado or self._pendiente is not None)
                    if self._cerrado:
                        return
                    imagen, offset, capturado = self._pendiente
                    self._pendiente = None
                resultado = self.detector.detectar(imagen, offset)
                with self._condicion:
                    self._resultado = (resultado, capturado)
        except Exception as error:
            with self._condicion:
                self._error = error

    def obtener(self):
        """Devuelve la última inferencia vigente o ausencia si está caducada."""
        with self._condicion:
            if self._error:
                raise RuntimeError("Falló el trabajador de señales CNN") from self._error
            if self._resultado and time.monotonic() - self._resultado[1] <= self.cfg.CNN_RESULTADO_MAX_EDAD:
                return self._resultado[0]
            # No es una inferencia negativa: tampoco rearma PARE.
            return ResultadoSenales(actualizacion=-1)

    def cerrar(self):
        """Descarta pendientes y solicita el cierre del hilo."""
        with self._condicion:
            self._cerrado = True
            self._pendiente = None
            self._condicion.notify_all()
        self._hilo.join(timeout=5)
