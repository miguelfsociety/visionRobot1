"""Programación por reloj monótono sin dormir el ciclo de visión."""

import config


class ProgramadorPulsos:
    """Limita avances por señal y giros por FRECUENCIA_ENVIO."""

    def __init__(self):
        self.ultimo_avance = None
        self.ultimo_envio = None
        self.ultima_accion = None

    def avance_disponible(self, ahora, intervalo):
        """Indica si cumplió el intervalo de avance, sin consumir un pulso."""
        return self.ultimo_avance is None or ahora - self.ultimo_avance >= intervalo

    def permitir(self, comando, ahora, intervalo):
        """Recibe comando y tiempo monótono; devuelve si debe despacharse."""
        detener = comando.accion == "DETENER"
        prioridad = detener and self.ultima_accion != "DETENER"
        listo = self.ultimo_envio is None or ahora - self.ultimo_envio >= 1 / config.FRECUENCIA_ENVIO
        if not prioridad and not listo:
            return False
        if comando.accion == "AVANZAR":
            if not self.avance_disponible(ahora, intervalo):
                return False
            self.ultimo_avance = ahora
        if detener:
            self.ultimo_avance = None
        self.ultimo_envio = ahora
        self.ultima_accion = comando.accion
        return True
