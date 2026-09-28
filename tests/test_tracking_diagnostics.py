"""El diagnóstico de pérdidas de tracking (Fase 5.1, Bloque 0).

Sin cámara: los cuadros son sintéticos y el reloj, números escritos a mano. Lo
que se prueba es que el reporte cuente lo que dice contar —qué es «dentro del
trazo», cuánto dura un hueco, qué es un duplicado— porque sobre esos números se
va a decidir si la solución es más luz o tolerancia a huecos.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence as AbcSequence
from pathlib import Path
from statistics import median

import pytest

from lsm.cli.demo import (
    Diagnostico,
    _build_parser,
    _diagnostico_pedido,
    anotar_evento,
    escribir_diagnostico,
    letras_dinamicas,
)
from lsm.config import Config
from lsm.segmentation import (
    LetterEmitted,
    RejectionReason,
    WindowDynamic,
    WindowRejected,
)
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.tracking_diagnostics import (
    Context,
    FrameRecord,
    GuidedSession,
    TrackingRecorder,
    analyze,
    classify_contexts,
    find_gaps,
    pearson,
    render_report,
    report_to_json,
)
from lsm.types import FrameSlot, InvalidFrame, InvalidReason, Prediction, RawFrame
from lsm.types import Sequence as FrameSequence

CONFIG = Config()
SIN_MANO = InvalidFrame(reason=InvalidReason.NO_HAND)
MINI_A = bytes([10] * 16)
MINI_B = bytes([200] * 16)
#: Más separación entre cuadros que la ventana previa al candidato.
LEJOS_MS = CONFIG.diagnostics.pre_candidate_ms + 100.0


def mano(x: float) -> RawFrame:
    return to_frame(translated(canonical_hand(), x, 400.0), width=1280, height=720)


def grabar(
    cuadros: AbcSequence[tuple[FrameSlot, str]],
    *,
    paso_ms: float = 50.0,
    luminancia: float = 0.5,
    miniaturas: list[bytes] | None = None,
) -> TrackingRecorder:
    """Una sesión: (cuadro, estado de la máquina al llegar), a `paso_ms` fijos."""
    recorder = TrackingRecorder(config=CONFIG)
    for i, (slot, estado) in enumerate(cuadros):
        recorder.observe(
            slot,
            wall_ms=1000.0 + i * paso_ms,
            detector_ms=i * 33.0,
            luminance=luminancia,
            thumbnail=miniaturas[i] if miniaturas else b"",
            state=estado,
        )
    return recorder


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


def test_la_velocidad_solo_se_mide_entre_dos_cuadros_con_mano() -> None:
    recorder = grabar(
        [
            (mano(400.0), "TRACKING"),
            (mano(420.0), "TRACKING"),
            (SIN_MANO, "TRACKING"),
            (mano(440.0), "TRACKING"),
        ]
    )

    velocidades = [r.velocity for r in recorder.records]

    assert velocidades[0] is None
    assert velocidades[1] == pytest.approx(0.2, rel=0.05)  # 20 px / ~100 px
    assert velocidades[2] is None
    assert velocidades[3] is None  # tras un hueco no hay par que medir


def test_el_reloj_real_empieza_en_cero() -> None:
    recorder = grabar([(mano(400.0), "IDLE"), (mano(400.0), "TRACKING")], paso_ms=40.0)

    assert [r.wall_ms for r in recorder.records] == [0.0, 40.0]


def test_un_cuadro_repetido_por_la_camara_es_duplicado() -> None:
    recorder = grabar(
        [(mano(400.0), "TRACKING")] * 3, miniaturas=[MINI_A, MINI_A, MINI_B]
    )

    assert [r.duplicate for r in recorder.records] == [False, True, False]
    assert recorder.records[1].thumbnail_diff == 0.0
    assert recorder.records[2].thumbnail_diff == pytest.approx(190 / 255)


# --------------------------------------------------------------------------- #
# Contexto y huecos
# --------------------------------------------------------------------------- #


def test_un_hueco_que_empieza_en_el_candidato_cuenta_como_dinamico() -> None:
    """Es el caso que hoy descarta la J entera: la pérdida a mitad del trazo."""
    recorder = grabar(
        [
            (mano(400.0), "TRACKING"),
            (mano(400.0), "TRACKING"),  # quieta
            (mano(450.0), "DYNAMIC_CANDIDATE"),
            (SIN_MANO, "DYNAMIC_CANDIDATE"),
            (SIN_MANO, "TRACKING"),
            (mano(500.0), "TRACKING"),
        ],
        # Separados más que `pre_candidate_ms`: si no, los cuadros quietos caen
        # en la ventana previa a la entrada y cuentan, con razón, como trazo.
        paso_ms=LEJOS_MS,
    )

    contextos = classify_contexts(recorder.records, CONFIG)

    assert contextos[1] is Context.QUIET
    assert contextos[2:5] == (Context.DYNAMIC,) * 3


def test_los_cuadros_justo_antes_de_entrar_al_candidato_son_del_trazo() -> None:
    """El candidato nace a mitad del movimiento; su principio va antes."""
    pre = CONFIG.diagnostics.pre_candidate_ms
    paso = pre / 3.0
    recorder = grabar(
        [(mano(400.0 + 30.0 * i), "TRACKING") for i in range(8)]
        + [(mano(700.0), "DYNAMIC_CANDIDATE")],
        paso_ms=paso,
    )

    contextos = classify_contexts(recorder.records, CONFIG)

    assert contextos[-4:] == (Context.DYNAMIC,) * 4
    assert contextos[1] is Context.OTHER


def test_un_hueco_con_la_mano_quieta_hereda_quieta() -> None:
    recorder = grabar(
        [
            (mano(400.0), "STABLE"),
            (mano(400.0), "STABLE"),
            (SIN_MANO, "STABLE"),
            (mano(400.0), "TRACKING"),
        ]
    )

    huecos = find_gaps(recorder.records, classify_contexts(recorder.records, CONFIG))

    assert len(huecos) == 1
    assert huecos[0].context is Context.QUIET
    assert huecos[0].frames == 1
    assert huecos[0].duration_ms == 50.0


def test_un_hueco_que_sigue_abierto_al_final_no_tiene_duracion() -> None:
    recorder = grabar([(mano(400.0), "TRACKING"), (SIN_MANO, "TRACKING")])

    assert (
        find_gaps(recorder.records, classify_contexts(recorder.records, CONFIG)) == ()
    )


# --------------------------------------------------------------------------- #
# El reporte
# --------------------------------------------------------------------------- #


def _sesion_con_perdidas_rapidas() -> TrackingRecorder:
    """Pierde la mano siempre tras un cuadro rápido y nunca tras uno quieto."""
    cuadros: list[tuple[FrameSlot, str]] = []
    x = 400.0
    for _ in range(6):
        cuadros += [(mano(x), "TRACKING"), (mano(x), "TRACKING")]  # quieta
        x += 40.0
        cuadros += [(mano(x), "DYNAMIC_CANDIDATE"), (SIN_MANO, "DYNAMIC_CANDIDATE")]
    return grabar(cuadros, paso_ms=LEJOS_MS)


def test_las_perdidas_tras_cuadros_rapidos_se_correlacionan_con_la_velocidad() -> None:
    recorder = _sesion_con_perdidas_rapidas()

    reporte = analyze(recorder.records, recorder.events, CONFIG)

    assert reporte.velocity_correlation is not None
    assert reporte.velocity_correlation > 0.9
    assert reporte.rates[Context.QUIET].rate == 1.0
    assert reporte.rates[Context.DYNAMIC].rate == pytest.approx(0.5)


def test_los_fps_sin_duplicados_descuentan_los_repetidos() -> None:
    recorder = grabar(
        [(mano(400.0), "TRACKING")] * 5,
        paso_ms=50.0,
        miniaturas=[MINI_A, MINI_A, MINI_B, MINI_B, MINI_A],
    )

    reporte = analyze(recorder.records, recorder.events, CONFIG)

    assert reporte.fps == pytest.approx(20.0)
    assert reporte.fps_unique == pytest.approx(10.0)
    assert reporte.wall_interval_ms == 50.0
    assert reporte.detector_interval_ms == 33.0


def test_pearson_de_una_serie_constante_no_existe() -> None:
    assert pearson([1.0, 1.0, 1.0], [0.0, 1.0, 0.0]) is None


def test_el_reporte_se_escribe_y_se_serializa() -> None:
    recorder = _sesion_con_perdidas_rapidas()
    reporte = analyze(recorder.records, recorder.events, CONFIG)
    metadata = {"iluminacion": "habitual", "pre_candidate_ms": 300.0}

    texto = render_report(reporte, metadata)
    datos = json.loads(
        json.dumps(report_to_json(reporte, recorder.records, [], metadata))
    )

    for seccion in ("## 1.", "## 2. Huecos", "## 3.", "## 4. Tasa real"):
        assert seccion in texto
    assert datos["summary"]["detection_rate"]["QUIETA"] == 1.0
    assert len(datos["records"]) == len(recorder.records)


# --------------------------------------------------------------------------- #
# Sesión guiada
# --------------------------------------------------------------------------- #


def test_la_sesion_guiada_pide_cada_letra_n_veces_en_orden() -> None:
    guiada = GuidedSession(letters=("J", "Z"), repetitions=2)
    vistas = []
    while not guiada.finished:
        vistas.append(guiada.current())
        guiada.advance()

    assert vistas == [("J", 1), ("J", 2), ("Z", 1), ("Z", 2)]
    assert guiada.current() is None


def test_backspace_vuelve_a_pedir_la_repeticion_anterior() -> None:
    guiada = GuidedSession(letters=("J",), repetitions=3)
    guiada.advance()
    guiada.advance()

    guiada.discard_last()

    assert guiada.current() == ("J", 2)
    assert guiada.discarded == [("J", 2)]


def test_las_letras_del_diagnostico_son_las_ocho_dinamicas_en_orden() -> None:
    assert letras_dinamicas() == (
        "J",
        "K",
        "DOBLE_L",
        "ENIE",
        "Q",
        "DOBLE_R",
        "X",
        "Z",
    )


def test_el_reporte_de_la_sesion_guiada_va_por_repeticion() -> None:
    recorder = TrackingRecorder(config=CONFIG)
    cuadros: list[tuple[FrameSlot, str, str, int]] = [
        (mano(400.0), "TRACKING", "J", 1),
        (mano(450.0), "DYNAMIC_CANDIDATE", "J", 1),
        (SIN_MANO, "DYNAMIC_CANDIDATE", "J", 1),
        (mano(500.0), "TRACKING", "J", 2),
    ]
    for i, (slot, estado, letra, n) in enumerate(cuadros):
        recorder.observe(
            slot,
            wall_ms=i * 50.0,
            detector_ms=i * 33.0,
            luminance=0.4,
            thumbnail=b"",
            state=estado,
            prompt=letra,
            repetition=n,
        )
    diagnostico = Diagnostico(etiqueta="x", salida=Path("."), recorder=recorder)
    anotar_evento(
        diagnostico,
        WindowRejected(frame_index=2, reason=RejectionReason.DYNAMIC_INTERRUPTED),
    )
    anotar_evento(
        diagnostico,
        WindowDynamic(frame_index=3, window=FrameSequence(frames=(mano(1.0),))),
    )
    anotar_evento(
        diagnostico,
        LetterEmitted(
            frame_index=3,
            prediction=Prediction(label="J", confidence=0.9),
            window=FrameSequence(frames=(mano(1.0),)),
        ),
    )

    filas = analyze(recorder.records, recorder.events, CONFIG).repetitions

    assert [(f.prompt, f.repetition) for f in filas] == [("J", 1), ("J", 2)]
    assert filas[0].interrupted == 1
    assert filas[0].gaps_in_stroke == 1
    assert filas[1].strokes == 1
    assert filas[1].emitted == ("J",)


# --------------------------------------------------------------------------- #
# El CLI
# --------------------------------------------------------------------------- #


def _args(*argv: str) -> argparse.Namespace:
    return _build_parser().parse_args(list(argv))


def test_diagnosticar_arma_la_sesion_guiada_con_las_repeticiones_de_config() -> None:
    diagnostico = _diagnostico_pedido(
        _args("diagnosticar", "--iluminacion", "lampara"), CONFIG
    )

    assert isinstance(diagnostico, Diagnostico)
    assert diagnostico.guiada is not None
    assert diagnostico.guiada.repetitions == CONFIG.diagnostics.repetitions_per_letter
    assert diagnostico.etiqueta == "lampara"


def test_sin_diagnostico_no_se_registra_nada() -> None:
    assert _diagnostico_pedido(_args(), CONFIG) is None


@pytest.mark.parametrize(
    "argv",
    [
        ("--diagnostico", "habitual", "--desde-dataset", "data/raw"),
        ("--diagnostico", "habitual", "--medir-fps"),
        ("--diagnostico", "x", "diagnosticar", "--iluminacion", "y"),
        ("--diagnostico", "///"),
    ],
)
def test_combinaciones_sin_sentido_se_rechazan(argv: tuple[str, ...]) -> None:
    assert isinstance(_diagnostico_pedido(_args(*argv), CONFIG), str)


def test_escribe_reporte_registros_y_flujo(tmp_path: Path) -> None:
    recorder = _sesion_con_perdidas_rapidas()
    diagnostico = Diagnostico(
        etiqueta="habitual",
        salida=tmp_path,
        recorder=recorder,
        flujo=[mano(400.0), SIN_MANO],
    )

    carpeta = escribir_diagnostico(diagnostico, CONFIG, {"camara (backend)": "PRUEBA"})

    assert {p.name for p in carpeta.iterdir()} == {
        "diagnostico.md",
        "diagnostico.json",
        "flujo.json",
    }
    datos = json.loads((carpeta / "diagnostico.json").read_text(encoding="utf-8"))
    assert datos["metadata"]["hands.min_tracking_confidence"] == 0.5
    assert datos["metadata"]["camara (backend)"] == "PRUEBA"


# --------------------------------------------------------------------------- #
# El bucle en vivo, con cámara, detector y OpenCV de mentira
# --------------------------------------------------------------------------- #


class _CamaraFalsa:
    """Entrega cuadros vacíos con luminancia y miniatura; la mano la pone el
    detector falso. Cada cuadro par repite la miniatura del anterior."""

    def __init__(self) -> None:
        self.leidos = 0
        #: Reloj de la sesión: avanza un cuadro a 30 fps por cada lectura, así
        #: la tasa medida al arrancar es 30 y los umbrales en milisegundos se
        #: traducen a los cuadros de siempre.
        self.reloj = 0.0

    def __enter__(self) -> _CamaraFalsa:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> object:
        from lsm.io.camera import CameraFrame

        self.leidos += 1
        self.reloj += 1.0 / 30.0
        return CameraFrame(
            bgr=None,
            rgb=None,
            width=1280,
            height=720,
            mean_luminance=0.3,
            thumbnail=bytes([self.leidos // 2 % 250] * 16),
        )

    def backend_name(self) -> str:
        return "FALSA"


class _DetectorFalso:
    def __init__(self, slots: list[FrameSlot]) -> None:
        self.slots = slots
        self.timestamp_ms = 0

    def __enter__(self) -> _DetectorFalso:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def detect(self, image: object) -> FrameSlot:  # noqa: ARG002
        self.timestamp_ms += 33
        return self.slots.pop(0) if self.slots else SIN_MANO


def test_una_sesion_guiada_en_vivo_escribe_su_diagnostico(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El cableado entero del bucle en vivo, sin hardware: calentamiento, un
    trazo con un hueco en medio, ESPACIO para dar la repetición por hecha y `q`
    para salir. Lo que tiene que quedar escrito es el diagnóstico."""
    import types

    import lsm.cli.demo as demo
    import lsm.io.camera as camera_io
    import lsm.io.preview as preview
    from lsm.classifiers.registry import ClassifierRegistry

    calentamiento = CONFIG.telemetry.fps_window_frames
    quieta = [mano(400.0)] * 12
    trazo = [mano(400.0 + 25.0 * i) for i in range(1, 16)]
    con_hueco: list[FrameSlot] = [*trazo[:12], SIN_MANO, *trazo[12:]]
    slots: list[FrameSlot] = [
        *([mano(400.0)] * calentamiento),
        *quieta,
        *con_hueco,
        *([trazo[-1]] * 20),
    ]
    # El calentamiento lee además `warmup_discard_frames` cuadros de la cámara
    # sin pasarlos al detector, y cada uno consume una tecla.
    descartados = CONFIG.telemetry.warmup_discard_frames
    teclas = [255] * (descartados + len(slots) - 1)
    teclas[descartados + calentamiento + 20] = 32  # ESPACIO: repetición hecha
    teclas.append(ord("q"))

    cv2_falso = types.SimpleNamespace(
        flip=lambda imagen, _eje: imagen,
        imshow=lambda *_a: None,
        waitKey=lambda _ms: teclas.pop(0) if teclas else ord("q"),
        destroyAllWindows=lambda: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "cv2", cv2_falso)
    monkeypatch.setattr(preview, "draw_demo_hud", lambda *_a: None)
    monkeypatch.setattr(preview, "draw_landmarks", lambda *_a, **_k: None)
    camara = _CamaraFalsa()
    monkeypatch.setattr(camera_io.Camera, "from_config", lambda *_a, **_k: camara)
    monkeypatch.setattr(
        demo, "time", types.SimpleNamespace(perf_counter=lambda: camara.reloj)
    )
    total = len(slots)
    monkeypatch.setattr(demo, "build_detector", lambda _c, **_k: _DetectorFalso(slots))

    diagnostico = _diagnostico_pedido(
        _args(
            "--diagnostico-salida",
            str(tmp_path),
            "diagnosticar",
            "--iluminacion",
            "habitual",
            "--repeticiones",
            "2",
            "--sin-reposo",
        ),
        CONFIG,
    )
    assert isinstance(diagnostico, Diagnostico)

    codigo = demo._sesion_en_vivo(
        CONFIG,
        ClassifierRegistry(),
        demo.Sesion(config=CONFIG),
        diagnostico=diagnostico,
    )

    assert codigo == 0
    (carpeta,) = tmp_path.iterdir()
    datos = json.loads((carpeta / "diagnostico.json").read_text(encoding="utf-8"))
    registros = datos["records"]
    # El calentamiento no se registra: mide la tasa antes de que exista la
    # máquina de estados.
    assert len(registros) == total - calentamiento
    assert registros[0]["prompt"] == "J"
    assert registros[-1]["repetition"] == 2  # ESPACIO avanzó a la segunda
    assert any(e["kind"] == "DYNAMIC_INTERRUPTED" for e in datos["events"])
    assert datos["summary"]["duplicates"] > 0
    assert datos["metadata"]["camara (backend)"] == "FALSA"
    assert (carpeta / "flujo.json").exists()


def test_una_repeticion_rehecha_cuenta_como_dos_intentos() -> None:
    """J#1 descartada con BACKSPACE y vuelta a hacer: dos filas, no una suma."""
    recorder = TrackingRecorder(config=CONFIG)
    secuencia = [("J", 1)] * 3 + [("J", 2)] * 2 + [("J", 1)] * 4
    for i, (letra, n) in enumerate(secuencia):
        recorder.observe(
            mano(400.0),
            wall_ms=i * 50.0,
            detector_ms=i * 33.0,
            luminance=0.4,
            thumbnail=b"",
            state="TRACKING",
            prompt=letra,
            repetition=n,
        )

    filas = analyze(recorder.records, recorder.events, CONFIG).repetitions

    assert [(f.prompt, f.repetition, f.frames) for f in filas] == [
        ("J", 1, 3),
        ("J", 2, 2),
        ("J", 1, 4),
    ]


def test_sondeo_reporta_luminancia_media_y_la_muestra_en_la_tabla() -> None:
    from lsm.tracking_diagnostics import render_probes, summarize_probe

    probe = summarize_probe(
        "auto 1280x720 exposición -6.0",
        "auto 1280x720 @ 30",
        [0.0, 33.0, 66.0, 99.0],
        [b"a", b"b", b"b", b"c"],
        [0.2, 0.4, 0.4, 0.6],
    )
    assert probe.mean_luminance == pytest.approx(0.4)
    assert "0.40" in render_probes([probe], {})


def test_sondeo_sin_luminancias_no_inventa_una() -> None:
    from lsm.tracking_diagnostics import summarize_probe

    probe = summarize_probe("x", "y", [0.0, 33.0], [b"a", b"b"])
    assert probe.mean_luminance is None


def test_diagnosticar_empieza_por_las_posturas_de_reposo() -> None:
    diagnostico = _diagnostico_pedido(
        _args("diagnosticar", "--iluminacion", "x"), CONFIG
    )
    assert isinstance(diagnostico, Diagnostico)
    assert diagnostico.guiada is not None
    assert diagnostico.guiada.current() == ("REPOSO_ESTATICA", 1)
    solo = _diagnostico_pedido(
        _args("diagnosticar", "--iluminacion", "x", "--solo-reposo"), CONFIG
    )
    assert isinstance(solo, Diagnostico)
    assert solo.guiada is not None
    assert solo.guiada.letters == ()


def _graba(sesion: GuidedSession) -> bool:
    """Sin estrechar el tipo: recording cambia entre llamadas."""
    return sesion.recording


def test_el_reposo_arranca_con_espacio_y_avanza_solo_al_cumplir_su_duracion() -> None:
    from lsm.tracking_diagnostics import REST_POSES

    sesion = GuidedSession(
        letters=("J",), repetitions=1, rest_poses=tuple(REST_POSES), rest_ms=5000.0
    )
    assert not _graba(sesion)
    sesion.tick(99999.0)
    assert sesion.current() == ("REPOSO_ESTATICA", 1)
    sesion.press_next(1000.0)
    assert _graba(sesion)
    sesion.press_next(2000.0)  # un segundo ESPACIO no la corta
    sesion.tick(5999.0)
    assert sesion.current() == ("REPOSO_ESTATICA", 1)
    sesion.tick(6000.0)
    assert sesion.current() == ("REPOSO_J", 1)
    assert not _graba(sesion)
    sesion.press_next(7000.0)
    sesion.discard_last()  # BACKSPACE reinicia la postura en curso
    assert sesion.current() == ("REPOSO_J", 1)
    assert not _graba(sesion)
    sesion.press_next(8000.0)
    sesion.tick(13000.0)
    assert sesion.current() == ("J", 1)
    assert _graba(sesion)


def _grabar_reposo(paso_ms: float, temblor: float) -> list[FrameRecord]:
    """Mano quieta con un temblor POR CUADRO de amplitud fija, a la tasa dada."""
    recorder = TrackingRecorder(CONFIG)
    base = canonical_hand()
    azar = random.Random(0)
    for i in range(60):
        dx = azar.uniform(-temblor, temblor)
        recorder.observe(
            to_frame(translated(base, 600.0 + dx, 400.0), width=1280, height=720),
            wall_ms=i * paso_ms,
            detector_ms=i * paso_ms,
            luminance=0.3,
            thumbnail=bytes([i % 256]),
            state="IDLE",
            prompt="REPOSO_ESTATICA",
            repetition=1,
        )
    return recorder.records


def test_el_temblor_por_cuadro_crece_con_la_tasa_en_pares_y_no_en_la_ventana() -> None:
    from lsm.tracking_diagnostics import summarize_rest

    lento = summarize_rest(_grabar_reposo(1000.0 / 15, 3.0))[0]
    rapido = summarize_rest(_grabar_reposo(1000.0 / 30, 3.0))[0]
    # Mismo temblor por cuadro: por segundo, los pares lo duplican al doble de
    # tasa; contra el cuadro de hace ~100 ms, no.
    pares = median(rapido.pair_per_s) / median(lento.pair_per_s)
    ventana = median(rapido.window_per_s) / median(lento.window_per_s)
    assert pares == pytest.approx(2.0, rel=0.05)
    assert 0.6 < ventana < 1.5
    assert rapido.fps == pytest.approx(30.0)


def test_el_reporte_trae_la_prueba_de_reposo_y_no_la_cuenta_como_repeticion() -> None:
    records = _grabar_reposo(1000.0 / 30, 3.0)
    reporte = analyze(records, [], CONFIG)
    texto = render_report(
        reporte,
        {
            "segmentation.velocity_threshold_per_s": 0.6,
            "segmentation.motion_threshold_per_s": 0.75,
            "diagnostics.rest_velocity_window_ms": 100.0,
        },
    )
    assert "## 6. Prueba de reposo" in texto
    assert "REPOSO_ESTATICA" in texto
    assert reporte.repetitions == ()
    resumen = report_to_json(reporte, records, [], {})["summary"]
    assert resumen["rest"][0]["frames"] == 60
