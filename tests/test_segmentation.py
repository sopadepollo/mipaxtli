"""La máquina de estados de `ARQUITECTURA.md` §4.2.

El video es continuo y las letras son discretas. Sin esta máquina el sistema
emite 30 letras por segundo, o letras basura mientras la mano viaja de una
posición a otra.

Cada test ejercita una transición concreta con secuencias sintéticas: mano que
aparece, se detiene, se clasifica, entra en cooldown, tiembla, vuelve a moverse y
desaparece. Sin cámara y sin clasificador real.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from lsm.config import Config
from lsm.segmentation import (
    HandAcquired,
    HandLost,
    LetterEmitted,
    RejectionReason,
    SegmentationEvent,
    State,
    StateChanged,
    WindowRejected,
    WindowStable,
    run_segmentation,
)
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.types import FrameSlot, InvalidFrame, InvalidReason, Prediction, Sequence

Classify = Callable[[Sequence], Prediction]

#: Umbrales apretados para que las secuencias de prueba quepan en pocos frames.
CONFIG = Config.model_validate(
    {
        "segmentation": {
            "buffer_size": 6,
            "stable_frames": 2,
            "emit_cooldown_frames": 3,
            "reject_cooldown_frames": 2,
            "missing_frames_to_idle": 2,
            "min_confidence": 0.6,
            "velocity_threshold": 0.02,
        }
    }
)

CONFIDENT_A = Prediction(label="A", confidence=0.95)
UNSURE_A = Prediction(label="A", confidence=0.2)

FINGERTIPS = (4, 8, 12, 16, 20)


def always(prediction: Prediction) -> Classify:
    def classify(sequence: Sequence) -> Prediction:  # noqa: ARG001 — doble de pruebas
        return prediction

    return classify


def responses(*predictions: Prediction) -> Classify:
    """Devuelve las predicciones en orden; la última se repite indefinidamente."""
    pending = list(predictions)

    def classify(sequence: Sequence) -> Prediction:  # noqa: ARG001 — doble de pruebas
        return pending.pop(0) if len(pending) > 1 else pending[0]

    return classify


def still_frames(
    count: int, *, at: tuple[float, float] = (640.0, 400.0)
) -> list[FrameSlot]:
    """La mano quieta en el mismo sitio: velocidad cero."""
    frame = to_frame(translated(canonical_hand(), *at), width=1280, height=720)
    return [frame for _ in range(count)]


def moving_frames(count: int, *, step: float = 25.0) -> list[FrameSlot]:
    """La mano viajando por el encuadre: velocidad muy por encima del umbral."""
    return [
        to_frame(
            translated(canonical_hand(), 300.0 + step * index, 400.0),
            width=1280,
            height=720,
        )
        for index in range(count)
    ]


def jittery_frames(count: int) -> list[FrameSlot]:
    """La mano en el sitio pero temblando: velocidad baja, dispersión alta.

    Es el caso que el indicador σ existe para atrapar: la mano no viaja, así que
    la máquina la da por estable, pero los dedos todavía se están acomodando y el
    vector promediado no representa ninguna configuración real.
    """
    hand = translated(canonical_hand(), 640.0, 400.0)
    wobbled = tuple(
        (x + 2.0, y, z) if index in FINGERTIPS else (x, y, z)
        for index, (x, y, z) in enumerate(hand)
    )
    frames = [
        to_frame(hand, width=1280, height=720),
        to_frame(wobbled, width=1280, height=720),
    ]
    return [frames[index % 2] for index in range(count)]


def run(
    stream: Iterable[FrameSlot], classify: Classify | None = None
) -> list[SegmentationEvent]:
    return list(run_segmentation(stream, CONFIG, classify or always(CONFIDENT_A)))


def transitions(events: list[SegmentationEvent]) -> list[tuple[State, State]]:
    return [
        (event.previous, event.current)
        for event in events
        if isinstance(event, StateChanged)
    ]


def emitted(events: list[SegmentationEvent]) -> list[str]:
    return [
        event.prediction.label for event in events if isinstance(event, LetterEmitted)
    ]


# --------------------------------------------------------------------------- #
# Transiciones del diagrama
# --------------------------------------------------------------------------- #


def test_idle_a_tracking_cuando_aparece_una_mano() -> None:
    events = run(still_frames(1))

    assert any(isinstance(event, HandAcquired) for event in events)
    assert transitions(events)[0] == (State.IDLE, State.TRACKING)


def test_sin_mano_la_maquina_se_queda_en_idle() -> None:
    events = run(list(missing_frames(5)))

    assert transitions(events) == []
    assert emitted(events) == []


def test_tracking_a_stable_cuando_la_mano_se_detiene() -> None:
    events = run(still_frames(3))

    assert (State.TRACKING, State.STABLE) in transitions(events)
    assert any(isinstance(event, WindowStable) for event in events)


def test_una_mano_en_movimiento_nunca_llega_a_stable() -> None:
    """Es el caso que produciría letras basura durante el viaje entre señas."""
    events = run(moving_frames(12))

    assert (State.TRACKING, State.STABLE) not in transitions(events)
    assert emitted(events) == []


def test_stable_a_emit_con_confianza_suficiente() -> None:
    events = run(still_frames(3))

    assert emitted(events) == ["A"]
    assert (State.STABLE, State.EMIT) in transitions(events)


def test_emit_a_tracking_cuando_expira_el_cooldown() -> None:
    events = run(still_frames(3 + CONFIG.segmentation.emit_cooldown_frames))

    assert (State.EMIT, State.TRACKING) in transitions(events)


def test_tracking_a_idle_tras_perder_la_mano() -> None:
    events = run([*still_frames(2), *missing_frames(4)])

    assert any(isinstance(event, HandLost) for event in events)
    assert transitions(events)[-1] == (State.TRACKING, State.IDLE)


def test_stable_a_idle_tras_perder_la_mano() -> None:
    """Transición que el diagrama de §4.2 no dibuja pero que tiene que existir:
    la mano puede desaparecer con la ventana ya estable."""
    events = run([*still_frames(3), *missing_frames(4)], always(UNSURE_A))

    assert transitions(events)[-1] == (State.STABLE, State.IDLE)


def test_un_hueco_mas_corto_que_el_umbral_no_manda_a_idle() -> None:
    events = run([*still_frames(3), *missing_frames(1), *still_frames(1)])

    assert not any(isinstance(event, HandLost) for event in events)


def test_stable_a_tracking_cuando_la_mano_vuelve_a_moverse() -> None:
    """Otra transición implícita: si la mano arranca de nuevo, la ventana estable
    deja de serlo aunque todavía no se haya emitido nada."""
    events = run([*still_frames(3), *moving_frames(4)], always(UNSURE_A))

    assert (State.STABLE, State.TRACKING) in transitions(events)


# --------------------------------------------------------------------------- #
# Rechazo y cooldown
# --------------------------------------------------------------------------- #


def test_una_prediccion_insegura_no_escribe_nada() -> None:
    events = run(still_frames(6), always(UNSURE_A))

    assert emitted(events) == []
    assert any(
        isinstance(event, WindowRejected)
        and event.reason is RejectionReason.LOW_CONFIDENCE
        for event in events
    )


def test_unknown_no_escribe_nada() -> None:
    """Aunque venga con confianza alta: UNKNOWN es una clase de rechazo."""
    events = run(still_frames(6), always(Prediction.unknown(confidence=0.99)))

    assert emitted(events) == []


def test_un_rechazo_no_reclasifica_en_cada_frame() -> None:
    """Sin cooldown de rechazo la máquina se queda en STABLE quemando CPU:
    reclasifica la misma ventana 30 veces por segundo para volver a rechazarla."""
    calls = 0

    def counting(sequence: Sequence) -> Prediction:  # noqa: ARG001
        nonlocal calls
        calls += 1
        return UNSURE_A

    run(still_frames(14), counting)

    assert calls <= 5, f"la ventana se reclasificó {calls} veces"


def test_el_rechazo_cuesta_menos_que_la_emision() -> None:
    """Una letra que quedó apenas bajo el umbral no debe obligar a rehacer la
    seña completa: el cooldown de rechazo es más corto a propósito."""
    frames = still_frames(20)

    rejections = sum(
        1
        for event in run(frames, always(UNSURE_A))
        if isinstance(event, WindowRejected)
    )
    emissions = len(emitted(run(frames, always(CONFIDENT_A))))

    assert rejections > emissions


# --------------------------------------------------------------------------- #
# Regla de letras dobles
# --------------------------------------------------------------------------- #


def test_sostener_la_sena_no_repite_la_letra() -> None:
    """Sin el cerrojo de repetición, mantener la mano quieta veinte frames escribe
    "AAAA": la ventana se reestabiliza tras cada cooldown y vuelve a emitir."""
    events = run(still_frames(20))

    assert emitted(events) == ["A"]


def test_una_letra_distinta_sale_sin_pedir_rebote() -> None:
    """Deletrear "casa" no puede exigir sacudir la mano entre cada par de letras."""
    events = run(still_frames(20), responses(CONFIDENT_A, Prediction("B", 0.95)))

    assert emitted(events) == ["A", "B"]


def test_un_rebote_permite_la_letra_doble() -> None:
    """ "carro" y "llave" necesitan emitir dos veces seguidas la misma letra."""
    events = run([*still_frames(3), *moving_frames(3), *still_frames(6)])

    assert emitted(events) == ["A", "A"]


def test_perder_la_mano_tambien_libera_el_cerrojo() -> None:
    events = run([*still_frames(3), *missing_frames(3), *still_frames(4)])

    assert emitted(events) == ["A", "A"]


def test_la_repeticion_bloqueada_se_reporta_con_su_propio_motivo() -> None:
    """No es lo mismo que una confianza baja: el clasificador acertó, lo que falta
    es el rebote. Quien depure la demo necesita poder distinguirlos."""
    events = run(still_frames(20))

    assert any(
        isinstance(event, WindowRejected)
        and event.reason is RejectionReason.REPEATED_LETTER
        for event in events
    )


def test_el_rebote_dentro_del_cooldown_tambien_cuenta() -> None:
    """El movimiento se evalúa en todos los estados, también durante EMIT.

    Si el rebote cayera entero dentro del cooldown y no se mirara, el cerrojo no se
    liberaría y la segunda letra quedaría bloqueada sin que quien firma pueda hacer
    nada al respecto.
    """
    cooldown = CONFIG.segmentation.emit_cooldown_frames
    stream = [*still_frames(3), *moving_frames(cooldown), *still_frames(4)]

    events = run(stream)

    assert emitted(events) == ["A", "A"]


def test_una_ventana_inestable_se_rechaza_antes_de_clasificar() -> None:
    """σ por encima de `quality.max_dispersion`: la ventana no llega al modelo."""
    config = Config.model_validate(
        {
            "quality": {"max_dispersion": 1e-4},
            "segmentation": {
                "buffer_size": 6,
                "stable_frames": 2,
                "emit_cooldown_frames": 3,
                "reject_cooldown_frames": 2,
                "missing_frames_to_idle": 2,
            },
        }
    )

    def never_called(sequence: Sequence) -> Prediction:  # noqa: ARG001
        pytest.fail("no debe clasificarse una ventana inestable")

    events = list(run_segmentation(jittery_frames(8), config, never_called))

    assert any(
        isinstance(event, WindowRejected)
        and event.reason is RejectionReason.UNSTABLE_WINDOW
        for event in events
    )


def test_la_misma_ventana_temblorosa_pasa_con_el_umbral_por_defecto() -> None:
    """Control: el rechazo anterior viene del umbral, no de que el temblor rompa
    la tubería."""
    events = run(jittery_frames(6))

    assert emitted(events) == ["A"]


# --------------------------------------------------------------------------- #
# Buffer y frames inválidos
# --------------------------------------------------------------------------- #


def test_la_ventana_emitida_no_excede_el_buffer() -> None:
    windows = [
        event.window
        for event in run(still_frames(30))
        if isinstance(event, WindowStable)
    ]

    assert windows
    assert all(len(window) <= CONFIG.segmentation.buffer_size for window in windows)


def test_un_frame_invalido_no_cose_dos_tramos_de_secuencia() -> None:
    """§0.3: los frames inválidos interrumpen la secuencia, no se interpolan.

    Si el buffer no se vaciara, la ventana mezclaría la mano de antes y de después
    del hueco e inventaría un movimiento que nadie ejecutó.
    """
    stream = [
        *still_frames(3, at=(200.0, 400.0)),
        *missing_frames(1),
        *still_frames(6, at=(900.0, 400.0)),
    ]

    windows = [event.window for event in run(stream) if isinstance(event, WindowStable)]

    assert len(windows) >= 2
    for window in windows:
        positions = {frame.landmarks[0].x for frame in window.frames}
        assert len(positions) == 1, "la ventana cruzó el hueco"


def test_un_frame_por_debajo_del_score_minimo_cuenta_como_ausencia() -> None:
    weak = to_frame(
        translated(canonical_hand(), 640.0, 400.0),
        width=1280,
        height=720,
        detection_score=0.1,
    )
    events = run([*still_frames(3), weak, weak, weak])

    assert any(isinstance(event, HandLost) for event in events)


def test_la_ventana_que_viaja_al_clasificador_es_una_secuencia() -> None:
    windows = [
        event.window
        for event in run(still_frames(4))
        if isinstance(event, WindowStable)
    ]

    assert windows
    assert all(isinstance(window, Sequence) for window in windows)


def test_los_eventos_son_tipados_no_cadenas() -> None:
    """El consumidor hace `match` sobre tipos, no compara cadenas."""
    events = run(still_frames(6))

    assert events
    for event in events:
        assert isinstance(
            event,
            HandAcquired
            | HandLost
            | StateChanged
            | WindowStable
            | WindowRejected
            | LetterEmitted,
        )
        assert event.frame_index >= 0


def missing_frames(count: int) -> list[FrameSlot]:
    return [InvalidFrame(reason=InvalidReason.NO_HAND) for _ in range(count)]
