"""Instrumentación de latencia: la aritmética, sin cámara y sin reloj real.

Todos los umbrales de la segmentación están expresados en **frames**, así que
«24 frames» significa 800 ms a 30 fps y 2 segundos a 12. Sin medir la tasa no se
puede calibrar ninguno, y la media sola no sirve para medirla: un pipeline que
promedia 28 fps pero cae a 9 durante medio segundo produce exactamente la
lentitud que se percibe y no se ve en el promedio. De ahí que lo que se fija aquí
sean los percentiles, no la media.

`lsm.telemetry` es código puro: recibe duraciones ya medidas y un reloj
inyectado. Eso es lo que permite probar el reparto por etapas con un reloj
guionizado en vez de dormir delante de una webcam.
"""

from __future__ import annotations

import pytest

from lsm.telemetry import (
    Cronometro,
    FrameTiming,
    Medicion,
    Stage,
    percentil,
    render_fps,
    render_resumen,
    resumir,
    segundos_restantes,
    tasa_insuficiente,
)


def _timing(
    camara: float = 0.004,
    deteccion: float = 0.020,
    segmentacion: float = 0.006,
    preview: float = 0.003,
) -> FrameTiming:
    return FrameTiming(
        camara=camara,
        deteccion=deteccion,
        segmentacion=segmentacion,
        preview=preview,
    )


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #


def test_el_percentil_interpola_linealmente_entre_los_rangos_vecinos() -> None:
    """La definición tiene que estar fijada, no heredada de una librería.

    Con cuatro valores, la mediana cae entre el segundo y el tercero: `q·(n−1)`
    da 1.5, y el resultado es el punto medio entre 20 y 30.
    """
    assert percentil([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0


def test_el_percentil_ordena_lo_que_recibe() -> None:
    """Quien mide no tiene por qué entregar la serie ordenada."""
    assert percentil([40.0, 10.0, 30.0, 20.0], 0.5) == 25.0


def test_el_percentil_de_un_solo_valor_es_ese_valor() -> None:
    for q in (0.0, 0.05, 0.5, 0.95, 1.0):
        assert percentil([7.5], q) == 7.5


def test_el_percentil_de_una_serie_vacia_es_un_error() -> None:
    """No hay valor razonable que devolver, y devolver 0.0 mentiría: 0 fps es un
    número que quien lea el reporte interpretaría como una caída."""
    with pytest.raises(ValueError, match="serie vacía"):
        percentil([], 0.5)


def test_el_percentil_fuera_de_cero_uno_es_un_error() -> None:
    with pytest.raises(ValueError, match="entre 0 y 1"):
        percentil([1.0, 2.0], 1.5)


# --------------------------------------------------------------------------- #
# El caso que motiva medir percentiles y no solo la media
# --------------------------------------------------------------------------- #


def test_la_media_esconde_la_caida_que_el_percentil_5_denuncia() -> None:
    """El caso exacto que hace inútil reportar solo la media.

    Cincuenta y cinco cuadros a 30 fps y cinco a 9: la media queda en 28.25 fps
    —un número que parece sano— mientras el percentil 5 dice 9. La lentitud que
    se percibe vive en esos cinco cuadros, así que el reporte tiene que
    enseñarlos.
    """
    serie = [30.0] * 55 + [9.0] * 5

    resumen = resumir(serie)

    assert resumen is not None
    assert resumen.media == pytest.approx(28.25)
    assert resumen.mediana == 30.0
    assert resumen.p5 == 9.0
    assert resumen.p95 == 30.0
    assert resumen.minimo == 9.0
    assert resumen.maximo == 30.0
    assert resumen.n == 60


def test_resumir_una_serie_vacia_da_nada() -> None:
    """Antes del primer cuadro no hay distribución, y eso no es un fallo: la
    demo arranca sin ninguna medida y el HUD tiene que poder decir `--`."""
    assert resumir([]) is None


# --------------------------------------------------------------------------- #
# El cuadro y sus etapas
# --------------------------------------------------------------------------- #


def test_el_ciclo_es_la_suma_de_las_cuatro_etapas() -> None:
    """Las cuatro etapas cubren el bucle entero por construcción, así que el
    ciclo es su suma y el fps de entrega es su inverso. Si alguna vez dejan de
    cubrirlo, este test es el que lo dice."""
    timing = _timing(camara=0.004, deteccion=0.020, segmentacion=0.006, preview=0.003)

    assert timing.ciclo == pytest.approx(0.033)


def test_el_procesamiento_excluye_la_espera_de_la_camara() -> None:
    """Son dos preguntas distintas y por eso se miden por separado.

    El fps de **entrega** es cuántos cuadros llegan por segundo: un techo que
    la cámara impone. El de **procesamiento** es cuántos podría sostener la
    tubería si la cámara entregara infinitamente rápido, y para eso hay que
    descontar el bloqueo de `camera.read()` — y también el preview, que es
    dibujo, no reconocimiento.
    """
    timing = _timing(camara=0.030, deteccion=0.020, segmentacion=0.006, preview=0.003)

    assert timing.procesamiento == pytest.approx(0.026)
    assert timing.ciclo == pytest.approx(0.059)


def test_una_etapa_negativa_es_un_error() -> None:
    """Un reloj que va hacia atrás es un fallo de instrumentación, no un dato:
    envenenaría la media sin que nada más lo denunciara."""
    with pytest.raises(ValueError, match="negativa"):
        _timing(deteccion=-0.001)


def test_las_etapas_se_recuperan_por_nombre() -> None:
    timing = _timing()

    por_etapa = timing.por_etapa()

    assert set(por_etapa) == set(Stage)
    assert por_etapa[Stage.DETECCION] == pytest.approx(0.020)


# --------------------------------------------------------------------------- #
# Acumulación
# --------------------------------------------------------------------------- #


def test_el_fps_de_entrega_es_cuadros_sobre_tiempo_de_pared() -> None:
    """No es la media de los inversos: es el caudal sostenido. Tres cuadros de
    33 ms son 90.9 fps sostenidos sobre 33 ms de pared, y ese es el número que
    responde «cuántas letras por segundo puede ver esto»."""
    medicion = Medicion()
    for _ in range(3):
        medicion.agregar(_timing())

    assert medicion.cuadros == 3
    assert medicion.segundos == pytest.approx(0.099)
    assert medicion.fps_entrega == pytest.approx(3 / 0.099)
    assert medicion.fps_procesamiento == pytest.approx(3 / 0.078)


def test_sin_cuadros_no_hay_fps_que_reportar() -> None:
    medicion = Medicion()

    assert medicion.cuadros == 0
    assert medicion.fps_entrega is None
    assert medicion.fps_procesamiento is None


def test_la_ventana_olvida_los_cuadros_viejos() -> None:
    """El fps que se muestra en vivo se promedia sobre los últimos cuadros y no
    sobre la sesión entera: un arranque lento no puede seguir tirando del número
    diez minutos después, porque entonces el HUD deja de servir para ver el
    efecto de lo que se acaba de tocar."""
    medicion = Medicion(ventana=2)

    medicion.agregar(_timing(camara=1.0))
    medicion.agregar(_timing())
    medicion.agregar(_timing())

    assert medicion.cuadros == 2
    assert medicion.segundos == pytest.approx(0.066)


def test_el_resumen_reparte_la_latencia_por_etapa_y_los_fps_en_percentiles() -> None:
    """Lo que el reporte tiene que contestar: a qué tasa corre esto, cuánto cae
    en el peor 5% y qué etapa se come el tiempo."""
    medicion = Medicion()
    for _ in range(18):
        medicion.agregar(_timing())
    # Dos cuadros lentos: MediaPipe se atragantó y el ciclo se fue a 100 ms. Son
    # el 10% de la serie, así que el percentil 5 cae dentro de ellos y los ve.
    for _ in range(2):
        medicion.agregar(_timing(deteccion=0.087))

    resumen = medicion.resumen()

    assert resumen.cuadros == 20
    assert resumen.entrega is not None
    assert resumen.entrega.mediana == pytest.approx(1 / 0.033)
    assert resumen.entrega.p5 == pytest.approx(10.0)
    assert resumen.etapas[Stage.DETECCION].maximo == pytest.approx(0.087)
    assert resumen.etapas[Stage.CAMARA].mediana == pytest.approx(0.004)
    assert resumen.fps_sostenido == pytest.approx(20 / medicion.segundos)


def test_el_resumen_de_una_medicion_vacia_no_inventa_distribuciones() -> None:
    resumen = Medicion().resumen()

    assert resumen.cuadros == 0
    assert resumen.entrega is None
    assert resumen.procesamiento is None
    assert resumen.etapas == {}
    assert resumen.fps_sostenido is None


def test_el_resumen_se_puede_volcar_como_texto() -> None:
    """El volcado es lo que se pega en la bitácora de la sesión, así que tiene
    que llevar las cuatro etapas y los percentiles con nombre."""
    medicion = Medicion()
    for _ in range(10):
        medicion.agregar(_timing())

    texto = render_resumen(medicion.resumen())

    for etapa in Stage:
        assert etapa.value in texto
    assert "p5" in texto
    assert "p95" in texto
    assert "mediana" in texto
    assert "cuadros: 10" in texto


# --------------------------------------------------------------------------- #
# El cronómetro
# --------------------------------------------------------------------------- #


def test_el_cronometro_reparte_las_marcas_entre_las_etapas() -> None:
    """El reloj entra inyectado para poder guionizarlo.

    Las marcas son las de un cuadro real del bucle en vivo: se abre, se lee la
    cámara, se detecta, se dibuja, y la segmentación se cierra al principio de
    la iteración siguiente porque es entonces cuando el consumidor devuelve el
    control al generador.
    """
    marcas = iter([0.0, 0.004, 0.024, 0.027, 0.033])
    crono = Cronometro(reloj=lambda: next(marcas))

    crono.iniciar()
    crono.marcar(Stage.CAMARA)
    crono.marcar(Stage.DETECCION)
    crono.marcar(Stage.PREVIEW)
    crono.marcar(Stage.SEGMENTACION)
    timing = crono.cerrar()

    assert timing.camara == pytest.approx(0.004)
    assert timing.deteccion == pytest.approx(0.020)
    assert timing.preview == pytest.approx(0.003)
    assert timing.segmentacion == pytest.approx(0.006)


def test_el_cronometro_sabe_si_hay_un_cuadro_a_medio_medir() -> None:
    """Es lo que el bucle consulta para no cerrar un cuadro que no empezó: el
    primero de la sesión no tiene iteración anterior que cerrar."""
    marcas = iter([0.0, 0.004])
    crono = Cronometro(reloj=lambda: next(marcas))

    assert not crono.abierto
    crono.iniciar()
    assert crono.abierto


def test_cerrar_sin_todas_las_etapas_es_un_error() -> None:
    """Un cuadro al que le falta una etapa daría un ciclo más corto que el real
    y un fps más alto que el real, que es la peor forma de equivocarse aquí."""
    marcas = iter([0.0, 0.004, 0.024])
    crono = Cronometro(reloj=lambda: next(marcas))

    crono.iniciar()
    crono.marcar(Stage.CAMARA)
    crono.marcar(Stage.DETECCION)

    with pytest.raises(ValueError, match="preview"):
        crono.cerrar()


def test_marcar_dos_veces_la_misma_etapa_es_un_error() -> None:
    """Sobrescribir la marca perdería el tiempo intermedio en silencio, y con
    ello la propiedad que sostiene todo el reporte: que el ciclo es la suma de
    las cuatro etapas y por tanto el tiempo de pared."""
    marcas = iter([0.0, 0.004, 0.024])
    crono = Cronometro(reloj=lambda: next(marcas))

    crono.iniciar()
    crono.marcar(Stage.CAMARA)

    with pytest.raises(ValueError, match="camara"):
        crono.marcar(Stage.CAMARA)


def test_marcar_sin_iniciar_es_un_error() -> None:
    crono = Cronometro(reloj=lambda: 0.0)

    with pytest.raises(ValueError, match="iniciar"):
        crono.marcar(Stage.CAMARA)


def test_el_cronometro_se_reutiliza_cuadro_a_cuadro() -> None:
    """Un solo cronómetro para toda la sesión: `iniciar` limpia las etapas del
    cuadro anterior en vez de acumularlas."""
    marcas = iter([0.0, 0.004, 0.024, 0.027, 0.033, 1.0, 1.008, 1.028, 1.031, 1.037])
    crono = Cronometro(reloj=lambda: next(marcas))

    for _ in range(2):
        crono.iniciar()
        crono.marcar(Stage.CAMARA)
        crono.marcar(Stage.DETECCION)
        crono.marcar(Stage.PREVIEW)
        crono.marcar(Stage.SEGMENTACION)
        timing = crono.cerrar()

    assert timing.camara == pytest.approx(0.008)
    assert not crono.abierto


# --------------------------------------------------------------------------- #
# La medición de duración fija
# --------------------------------------------------------------------------- #


def test_lo_que_falta_de_medicion_nunca_es_negativo() -> None:
    """Alimenta la cuenta atrás del HUD y la condición de parada del bucle. Un
    número negativo en pantalla sería ruido."""
    assert segundos_restantes(duracion=60.0, transcurrido=0.0) == 60.0
    assert segundos_restantes(duracion=60.0, transcurrido=59.5) == pytest.approx(0.5)
    assert segundos_restantes(duracion=60.0, transcurrido=60.0) == 0.0
    assert segundos_restantes(duracion=60.0, transcurrido=61.0) == 0.0


# --------------------------------------------------------------------------- #
# La linea del HUD
# --------------------------------------------------------------------------- #


def test_el_hud_dice_las_dos_tasas() -> None:
    """Las dos, no solo una: con `fps` bajo y `proc` alto el bucle espera a la
    camara, y con los dos bajos no llega la tuberia. Son diagnosticos distintos
    y en pantalla tienen que poder distinguirse."""
    assert render_fps(entrega=27.4, procesamiento=41.2) == "fps 27.4  proc 41.2"


def test_el_hud_admite_no_haber_medido_todavia() -> None:
    """El primer cuadro se dibuja antes de que exista ninguna medida: la
    segmentacion de ese cuadro aun no ha corrido."""
    assert render_fps(entrega=None, procesamiento=None) == "fps --  proc --"


# --------------------------------------------------------------------------- #
# El aviso de tasa baja
# --------------------------------------------------------------------------- #


def test_una_tasa_por_debajo_del_minimo_se_denuncia() -> None:
    """A tasas muy bajas ninguna ventana temporal razonable contiene frames
    suficientes, y quien esta delante de la camara tiene que saber que lo que
    falla es la maquina y no su sena."""
    assert tasa_insuficiente(9.0, minimo=12.0)
    assert not tasa_insuficiente(18.0, minimo=12.0)


def test_la_tasa_justo_en_el_minimo_no_avisa() -> None:
    """El minimo es el ultimo valor aceptable, no el primero que avisa."""
    assert not tasa_insuficiente(12.0, minimo=12.0)


def test_sin_medida_no_hay_nada_que_denunciar() -> None:
    """Antes del primer cuadro no se sabe la tasa. Avisar ahi seria avisar
    siempre, durante el primer segundo de cada sesion."""
    assert not tasa_insuficiente(None, minimo=12.0)
