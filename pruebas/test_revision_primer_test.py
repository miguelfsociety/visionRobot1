"""Regresiones de la primera prueba física: perfiles, fondos y telemetría."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import config
from comunicacion.salida_con_registro import SalidaConRegistro
from control.contratos import ComandoRobot, ResultadoLinea, ResultadoSenales, calcular_accion
from control.controlador import ControladorZonas
from control.estados import MaquinaEstados
from control.pulsos import ProgramadorPulsos
from utils.metricas import Cronometro, MetricasCiclo
from vision.propuestas_senales import proponer_senales


class PruebasPerfiles(unittest.TestCase):
    def contar(self, perfil, senal):
        maquina = MaquinaEstados(perfil=perfil)
        pulsos = ProgramadorPulsos()
        enviados = []
        for tick in range(640):
            tiempo = tick / 64
            comando = maquina.actualizar(ResultadoLinea(valida=True), ResultadoSenales(senal=senal, en_disparo=True), tiempo)
            if pulsos.permitir(comando, tiempo, comando.intervalo_avance):
                enviados.append(tiempo)
        return enviados

    def test_perfil_agil_cuadruplica_pulsos_rectos(self):
        for senal, esperados in (("SIGA", 40), ("VELOCIDAD_60", 80), ("ZONA_ESCOLAR", 20)):
            with self.subTest(senal=senal):
                original = self.contar("original", senal)
                agil = self.contar("agil", senal)
                self.assertEqual(len(agil), esperados)
                self.assertEqual(len(agil), len(original) * 4)
                self.assertTrue(all(b-a >= 1/config.FRECUENCIA_ENVIO for a,b in zip(agil,agil[1:])))

    def test_perfil_agil_conserva_intervalo_despues_de_pare(self):
        maquina = MaquinaEstados(perfil="agil")
        linea = ResultadoLinea(valida=True)
        maquina.actualizar(linea, ResultadoSenales(senal="ZONA_ESCOLAR", en_disparo=True), 0)
        pare = ResultadoSenales(senal="PARE", en_disparo=True)
        self.assertEqual(maquina.actualizar(linea, pare, 1).accion, "DETENER")
        self.assertEqual(maquina.actualizar(linea, pare, 3.99).accion, "DETENER")
        comando = maquina.actualizar(linea, pare, 4)
        self.assertEqual(comando.accion, "AVANZAR")
        self.assertEqual(comando.intervalo_avance, 0.5)

    def test_siga_se_registra_una_vez_por_aparicion(self):
        maquina = MaquinaEstados(perfil="agil")
        for i in range(100):
            maquina.actualizar(ResultadoLinea(valida=True), ResultadoSenales(senal="SIGA", en_disparo=True, actualizacion=i), i/64)
        eventos = [e for e in maquina.obtener_eventos() if e[2] == "SIGA"]
        self.assertEqual(len(eventos), 1)

    def test_perfil_desconocido_se_rechaza(self):
        with self.assertRaises(ValueError):
            MaquinaEstados(perfil="desconocido")

    def test_alternancia_giro_avance_independiente_de_fps(self):
        secuencias=[]
        for fps in (64,128,256):
            controlador=ControladorZonas(sincronizar_envios=True)
            pulsos=ProgramadorPulsos()
            linea=ResultadoLinea(valida=True,error=.25)
            for _ in range(40): controlador.calcular(linea)
            enviados=[]
            for tick in range(fps*4):
                ahora=tick/fps
                controlador.avance_habilitado=pulsos.avance_disponible(ahora,.125)
                izq,der=controlador.calcular(linea)
                comando=ComandoRobot(accion=calcular_accion(izq,der,config.DIF_GIRO))
                if pulsos.permitir(comando,ahora,.125):
                    enviados.append(comando.accion);controlador.confirmar_envio()
            self.assertIn("AVANZAR",enviados)
            self.assertIn("GIRAR_DERECHA",enviados)
            secuencias.append(enviados)
        self.assertEqual(secuencias[0],secuencias[1])
        self.assertEqual(secuencias[1],secuencias[2])

    def test_mantiene_giros_mientras_espera_intervalo_de_avance(self):
        controlador=ControladorZonas(sincronizar_envios=True)
        linea=ResultadoLinea(valida=True,error=.25)
        for _ in range(40): controlador.calcular(linea)
        controlador.confirmar_envio()  # Giro emitido; ahora el plan pide avance.
        controlador.avance_habilitado=False
        for _ in range(3):
            izq,der=controlador.calcular(linea)
            self.assertNotEqual(izq,der)
            controlador.confirmar_envio()
        controlador.avance_habilitado=True
        izq,der=controlador.calcular(linea)
        self.assertEqual(izq,der)


class PruebasPropuestas(unittest.TestCase):
    def test_triangulo_de_fondo_no_es_propuesta(self):
        imagen = np.full((200, 200, 3), 240, np.uint8)
        cv2.fillPoly(imagen, [np.array([[0,0],[100,0],[0,100]],np.int32)], (0,180,230))
        self.assertEqual(proponer_senales(imagen), [])

    def test_circulo_y_pentagono_siguen_admitidos(self):
        imagen = np.full((200, 400, 3), 240, np.uint8)
        cv2.circle(imagen,(90,100),40,(0,0,220),-1)
        cv2.fillPoly(imagen,[np.array([[270,50],[310,85],[300,140],[240,140],[230,85]],np.int32)],(0,180,230))
        self.assertEqual({p["color_propuesta"] for p in proponer_senales(imagen)}, {"rojo","amarillo"})


class PruebasRegistros(unittest.TestCase):
    def test_iniciar_no_consume_pare_durante_la_espera(self):
        import main
        reloj=[0.0]
        args=main.construir_parser().parse_args(["--video","falso.mp4","--detector","clasico",
            "--salida","mbot","--etapa","completo","--debug"])
        fuente=MagicMock()
        fuente.leer.side_effect=[(True,np.zeros((100,100,3),np.uint8),0),(True,np.zeros((100,100,3),np.uint8),4)]
        salida=MagicMock()
        teclas=iter([ord("i"),ord("q")])
        def tecla(*args):
            codigo=next(teclas)
            if codigo==ord("i"): reloj[0]=4.0
            return codigo
        with patch("main._crear_fuente_video",return_value=fuente),patch("main._crear_salida",return_value=salida), \
                patch("main.MetricasCiclo"),patch("main.time.monotonic",side_effect=lambda:reloj[0]), \
                patch("main.preprocesar",return_value=np.zeros((100,100,3),np.uint8)), \
                patch("main.recortar_zonas",return_value=(np.zeros((50,100,3),np.uint8),np.zeros((50,100,3),np.uint8),0,0)), \
                patch("main.detectar_linea",return_value=(ResultadoLinea(valida=True),np.zeros((50,100),np.uint8))), \
                patch("main.detectar_senales",return_value=ResultadoSenales(senal="PARE",en_disparo=True)), \
                patch("main.construir_mosaico",return_value=np.zeros((100,100,3),np.uint8)), \
                patch("main.cv2.namedWindow"),patch("main.cv2.imshow"),patch("main.cv2.destroyAllWindows"), \
                patch("main.cv2.waitKey",side_effect=tecla):
            main.ejecutar(args)
        self.assertEqual(salida.enviar.call_count,1)
        self.assertEqual(salida.enviar.call_args.args[0].accion,"DETENER")

    def test_registro_de_despachos_bluetooth_y_parada(self):
        class SalidaFalsa:
            def __init__(self): self.eventos=[]
            def enviar(self,c): self.eventos.append(c.accion)
            def detener(self): self.eventos.append("DETENER")
            def cerrar(self): pass
        with tempfile.TemporaryDirectory() as carpeta:
            real=SalidaFalsa()
            salida=SalidaConRegistro(real,Path(carpeta)/"comandos.csv")
            salida.enviar(ComandoRobot(accion="AVANZAR",senal="SIGA",intervalo_avance=.25))
            salida.detener();salida.cerrar()
            with (Path(carpeta)/"comandos.csv").open(encoding="utf-8") as f: filas=list(csv.DictReader(f))
            self.assertEqual([f["accion"] for f in filas],real.eventos)
            self.assertEqual(filas[0]["intervalo_avance"],"0.25")

    def test_telemetria_separa_ciclo_cnn_y_decision(self):
        with tempfile.TemporaryDirectory() as carpeta:
            metricas=MetricasCiclo(carpeta)
            cronometro=Cronometro();cronometro.iniciar_ciclo()
            metricas.registrar_fotograma(cronometro,ResultadoSenales(senal="PARE",latencia_inferencia_s=.2,edad_s=.3),ComandoRobot(),"agil")
            metricas.cerrar()
            with (Path(carpeta)/"metricas_fotogramas.csv").open(encoding="utf-8") as f: fila=next(csv.DictReader(f))
            self.assertEqual(float(fila["latencia_cnn_s"]),.2)
            self.assertEqual(fila["senal"],"PARE")
            self.assertEqual(fila["perfil"],"agil")


if __name__ == "__main__":
    unittest.main()
