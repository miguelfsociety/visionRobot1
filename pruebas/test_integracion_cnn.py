"""Pruebas de decisiones, concurrencia y datos sin robot físico."""

import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

import config
from comunicacion.salida import SalidaRegistro
from comunicacion.salida_asincrona import SalidaAsincrona
from control.contratos import ComandoRobot, ResultadoLinea, ResultadoSenales
from control.estados import MaquinaEstados, PARE
from control.pulsos import ProgramadorPulsos
from vision.senales_cnn import DetectorSenalesCNN, TrabajadorSenales

RAIZ = Path(__file__).resolve().parents[1]
LINEA = ResultadoLinea(valida=True, confianza=4)


class PruebasControlSenales(unittest.TestCase):
    def test_modos_sin_movimiento_no_abren_bluetooth(self):
        import main
        for etapa in ("vision", "comandos"):
            with self.subTest(etapa=etapa):
                args = main.construir_parser().parse_args(["--video", "inexistente.mp4", "--detector", "clasico",
                    "--salida", "mbot", "--etapa", etapa])
                fuente = MagicMock()
                fuente.leer.return_value = (False, None, 0)
                salida = MagicMock()
                with patch("main._crear_fuente_video", return_value=fuente), patch("main._crear_salida", return_value=salida) as crear, \
                        patch("main.MetricasCiclo"):
                    main.ejecutar(args)
                crear.assert_called_once_with("nula")
                salida.enviar.assert_not_called()
                fuente.liberar.assert_called_once()

    def test_intervalos_y_registro(self):
        for senal, intervalo in config.INTERVALOS_AVANCE.items():
            with self.subTest(senal=senal), tempfile.TemporaryDirectory() as carpeta:
                maquina, pulsos = MaquinaEstados(), ProgramadorPulsos()
                salida = SalidaRegistro(str(Path(carpeta) / "comandos.csv"))
                for ahora in (0.0, intervalo / 2, intervalo, intervalo * 1.5, intervalo * 2):
                    comando = maquina.actualizar(LINEA, ResultadoSenales(senal=senal, en_disparo=True), ahora)
                    if pulsos.permitir(comando, ahora, comando.intervalo_avance):
                        salida.enviar(comando)
                salida.cerrar()
                with (Path(carpeta) / "comandos.csv").open(encoding="utf-8") as archivo:
                    filas = list(csv.DictReader(archivo))
                self.assertEqual(len(filas), 3)
                self.assertTrue(all(float(f["intervalo_avance"]) == intervalo for f in filas))

    def test_giros_independientes_y_parada_prioritaria(self):
        pulsos = ProgramadorPulsos()
        avance = ComandoRobot(accion="AVANZAR")
        giro = ComandoRobot(accion="GIRAR_DERECHA")
        self.assertTrue(pulsos.permitir(avance, 0, 2))
        self.assertFalse(pulsos.permitir(avance, 0.5, 2))
        self.assertTrue(pulsos.permitir(giro, 0.5, 2))
        self.assertTrue(pulsos.permitir(ComandoRobot(), 0.501, 2))

    def test_pare_tres_segundos_conserva_velocidad_y_no_se_repite(self):
        maquina = MaquinaEstados()
        maquina.actualizar(LINEA, ResultadoSenales(senal="ZONA_ESCOLAR", en_disparo=True), 0)
        pare = ResultadoSenales(senal="PARE", en_disparo=True, pare_visible=True, actualizacion=1)
        self.assertEqual(maquina.actualizar(LINEA, pare, 1).accion, "DETENER")
        self.assertEqual(maquina.actualizar(LINEA, pare, 3.999).accion, "DETENER")
        self.assertEqual(maquina.actualizar(LINEA, pare, 4).accion, "AVANZAR")
        for tiempo in (5, 10, 20):
            self.assertNotEqual(maquina.actualizar(LINEA, pare, tiempo).estado, PARE)
        self.assertEqual(maquina.intervalo_avance, 2)
        for secuencia in range(2, 7):
            maquina.actualizar(LINEA, ResultadoSenales(actualizacion=secuencia, pare_visible=False), 20 + secuencia)
        self.assertEqual(maquina.actualizar(LINEA,
            ResultadoSenales(senal="PARE", en_disparo=True, actualizacion=7, pare_visible=True), 30).estado, PARE)

    def test_repetir_resultado_no_rearma_pare(self):
        maquina = MaquinaEstados()
        maquina.actualizar(LINEA, ResultadoSenales(senal="PARE", en_disparo=True), 0)
        for tiempo in range(20):
            maquina.actualizar(LINEA, ResultadoSenales(actualizacion=1, pare_visible=False), tiempo)
        self.assertFalse(maquina.pare_armado)

    def test_pare_prevalece_sobre_busqueda_de_linea(self):
        maquina = MaquinaEstados()
        maquina.actualizar(ResultadoLinea(), ResultadoSenales(), 0)
        comando = maquina.actualizar(ResultadoLinea(), ResultadoSenales(senal="PARE", en_disparo=True), 1)
        self.assertEqual(comando.accion, "DETENER")


class ClasificadorFalso:
    def __init__(self, nombre="PARE", confianza=0.95):
        self.nombre, self.confianza = nombre, confianza

    def predecir(self, imagen, orden_color):
        return {"nombre": self.nombre, "confianza": self.confianza}


class PruebasDetectorCNN(unittest.TestCase):
    @staticmethod
    def propuesta():
        return {"color": "rojo", "area": 400, "bounding_rect": (20, 70, 20, 20),
                "centroide": (30, 80), "recorte": np.zeros((20, 20, 3), np.uint8)}

    def test_confirma_tres_inferencias_y_desplaza_todas_las_coordenadas(self):
        detector = DetectorSenalesCNN(ClasificadorFalso())
        with patch("vision.senales_cnn.proponer_senales", side_effect=lambda *a: [self.propuesta()]):
            resultados = [detector.detectar(np.zeros((100, 100, 3), np.uint8), 50) for _ in range(3)]
        self.assertEqual([r.senal for r in resultados], [None, None, "PARE"])
        self.assertTrue(resultados[-1].en_disparo)
        self.assertEqual(resultados[-1].candidatos[0]["bounding_rect"][1], 120)

    def test_baja_confianza_y_ausencia_no_confirmadas(self):
        detector = DetectorSenalesCNN(ClasificadorFalso(confianza=0.79))
        with patch("vision.senales_cnn.proponer_senales", side_effect=lambda *a: [self.propuesta()]):
            for _ in range(8):
                self.assertIsNone(detector.detectar(np.zeros((100, 100, 3), np.uint8)).senal)
        with patch("vision.senales_cnn.proponer_senales", return_value=[]):
            self.assertEqual(detector.detectar(np.zeros((100, 100, 3), np.uint8)).candidatos, [])

    def test_trabajador_descarta_intermedios(self):
        iniciado, liberar, terminado = threading.Event(), threading.Event(), threading.Event()
        recibidos = []
        class DetectorBloqueado:
            def detectar(self, imagen, offset):
                recibidos.append(int(imagen[0, 0, 0]))
                if len(recibidos) == 1:
                    iniciado.set()
                    liberar.wait(5)
                else:
                    terminado.set()
                return ResultadoSenales(actualizacion=len(recibidos))
        trabajador = TrabajadorSenales(DetectorBloqueado())
        try:
            trabajador.enviar(np.zeros((2, 2, 3), np.uint8))
            self.assertTrue(iniciado.wait(3))
            for valor in range(1, 10):
                trabajador.enviar(np.full((2, 2, 3), valor, np.uint8))
            liberar.set()
            self.assertTrue(terminado.wait(3))
            self.assertEqual(recibidos, [0, 9])
        finally:
            liberar.set()
            trabajador.cerrar()


class PruebasPrioridadSalida(unittest.TestCase):
    def test_no_sale_movimiento_pendiente_despues_de_parar(self):
        inicio, liberar = threading.Event(), threading.Event()
        eventos = []
        class SalidaBloqueada:
            def enviar(self, comando):
                inicio.set()
                liberar.wait(3)
                eventos.append("avance")
            def detener(self):
                eventos.append("parar")
            def cerrar(self):
                pass
        salida = SalidaAsincrona(SalidaBloqueada())
        try:
            salida.enviar(ComandoRobot(accion="AVANZAR"))
            self.assertTrue(inicio.wait(3))
            salida.enviar(ComandoRobot(accion="AVANZAR"))
            paro = threading.Thread(target=salida.detener)
            paro.start()
            liberar.set()
            paro.join(3)
            self.assertFalse(paro.is_alive())
        finally:
            liberar.set()
            salida.cerrar()
        self.assertEqual(eventos, ["avance", "parar"])


class PruebasParticiones(unittest.TestCase):
    def test_datos_preparados_sin_fuga_y_conversion_heic(self):
        particiones = RAIZ / "datos_preparados/particiones.csv"
        if not particiones.exists():
            self.skipTest("Dataset local aún no preparado")
        with particiones.open(encoding="utf-8") as archivo:
            filas = list(csv.DictReader(archivo))
        for campo in ("sha256", "sha256_pixeles"):
            self.assertEqual(len(filas), len({f[campo] for f in filas}))
        self.assertEqual({f["clase"] for f in filas}, {"0", "1", "2", "3"})
        self.assertEqual({f["conjunto"] for f in filas}, {"train", "val", "test"})
        resumen = json.loads((particiones.parent / "resumen.json").read_text(encoding="utf-8"))
        originales_heic = sum(p.suffix.lower() == ".heic" for c in "0123" for p in (RAIZ.parent / "dataset" / c).glob("*"))
        if originales_heic:
            self.assertEqual(resumen["heic_convertidos"], originales_heic)


if __name__ == "__main__":
    unittest.main()
