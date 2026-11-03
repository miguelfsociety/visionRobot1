"""Registra los envíos completados incluso cuando se utiliza Bluetooth."""

import time

from comunicacion.salida import SalidaRobot, SalidaRegistro


class SalidaConRegistro(SalidaRobot):
    """Se coloca debajo de SalidaAsincrona para registrar el despacho real."""

    def __init__(self, salida_real, ruta):
        self.salida_real = salida_real
        self.registro = SalidaRegistro(ruta)

    def enviar(self, comando):
        inicio = time.monotonic()
        self.salida_real.enviar(comando)
        self.registro.enviar(comando, tiempo=inicio)

    def detener(self):
        self.salida_real.detener()
        self.registro.detener()

    def cerrar(self):
        try:
            self.salida_real.cerrar()
        finally:
            self.registro.cerrar()
