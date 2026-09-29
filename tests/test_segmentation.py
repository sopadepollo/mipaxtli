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
from dataclasses import replace

import pytest

from lsm.config import Config
from lsm.features import ExtractionRejected, extract_sequence_features
from lsm.gaps import GapPolicy
from lsm.segmentation import (
    EvidenceAccumulated,
    FrameThresholds,
    GapResolved,
    HandAcquired,
    HandLost,
    LetterEmitted,
    RejectionReason,
    SegmentationEvent,
    State,
    StateChanged,
    WindowDynamic,
    WindowRejected,
    WindowStable,
    frames_from_ms,
    run_segmentation,
)
from lsm.synthetic import canonical_hand, to_frame, translated
from lsm.types import (
    FrameSlot,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Prediction,
    RawFrame,
    Sequence,
    WindowOrigin,
)

Classify = Callable[[Sequence, WindowOrigin], Prediction]

#: La tasa con la que corren estos tests. Fija y explícita: los umbrales viven en
#: milisegundos y solo se convierten a cuadros con una tasa, así que sin fijarla
#: no habría nada determinista que afirmar.
FPS = 30.0


def _ms(frames: int) -> float:
    """Los milisegundos que son `frames` cuadros a `FPS`.

    Los tests de este archivo razonan en cuadros —«dos frames quieta y emite»—
    porque es como funciona la máquina. La configuración habla en milisegundos.
    Esto traduce entre las dos, en un solo sitio y de forma exacta a 30 fps.
    """
    return frames * 1000.0 / FPS


#: Umbrales apretados para que las secuencias de prueba quepan en pocos frames.
CONFIG = Config.model_validate(
    {
        # Los flujos de estos tests teletransportan la mano entre bloques (la
        # quieta en x=640 y el viaje desde x=300, o 700 px a través de un hueco
        # de un cuadro): el filtro de plausibilidad los invalida con razón
        # (§0.4). Estos tests son de la lógica de estados; el filtro tiene los
        # suyos en `tests/test_plausibility.py`.
        "plausibility": {"enabled": False},
        "segmentation": {
            "buffer_ms": _ms(6),
            "stable_ms": _ms(2),
            "emit_cooldown_ms": _ms(3),
            "reject_cooldown_ms": _ms(2),
            "missing_to_idle_ms": _ms(2),
            # Sin sostener huecos fuera del trazo (Paso 4): estos tests son de la
            # máquina de antes. El relleno ampliado tiene los suyos al final.
            "tracking_max_gap_ms": 0.0,
            "stable_max_gap_ms": 0.0,
            "min_confidence": 0.6,
            "high_confidence": 0.9,
            "velocity_threshold_per_s": 0.02 * FPS,
            # Un cuadro: la velocidad del §6.1 entre pares consecutivos. Estos
            # tests son de la lógica de la máquina; la ventana tiene los suyos.
            "velocity_window_ms": _ms(1),
            "closing_window_ms": _ms(1),
            # El camino dinámico, apartado de estos tests: el tránsito más largo
            # que usan son 12 frames de viaje, y 20 frames móviles no los
            # alcanza ninguno. Los tests del camino dinámico llevan su propia
            # configuración (`DYNAMIC_CONFIG`) y ejercitan la interacción a
            # propósito.
            "motion_min_ms": _ms(20),
            "motion_confirm_low_ms": _ms(4),
            "motion_max_ms": _ms(60),
        },
    }
)

#: Los mismos umbrales ya resueltos a cuadros, que es la unidad en la que se
#: escriben las expectativas de este archivo.
UMBRALES = FrameThresholds.from_config(CONFIG, FPS)

CONFIDENT_A = Prediction(label="A", confidence=0.95)
#: Por encima del piso de emision y por debajo del umbral alto: la ventana no
#: se rechaza, pero tampoco basta para emitir sin acumular mas evidencia.
MEDIUM_A = Prediction(label="A", confidence=0.7)
UNSURE_A = Prediction(label="A", confidence=0.2)

FINGERTIPS = (4, 8, 12, 16, 20)


def always(prediction: Prediction) -> Classify:
    def classify(sequence: Sequence, origin: WindowOrigin) -> Prediction:  # noqa: ARG001 — doble de pruebas
        return prediction

    return classify


def responses(*predictions: Prediction) -> Classify:
    """Devuelve las predicciones en orden; la última se repite indefinidamente."""
    pending = list(predictions)

    def classify(sequence: Sequence, origin: WindowOrigin) -> Prediction:  # noqa: ARG001 — doble de pruebas
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
    return list(
        run_segmentation(stream, CONFIG, classify or always(CONFIDENT_A), fps=FPS)
    )


def velocidades(window: Sequence) -> tuple[float, ...]:
    """Las velocidades de la §6 dentro de una ventana ya emitida.

    Es la forma de comprobar la invariante del ADR 0004 sobre la ventana que de
    verdad se clasificó, en vez de darla por cierta.
    """
    features = extract_sequence_features(window, CONFIG)
    assert not isinstance(features, ExtractionRejected)
    return features.velocities


def ventanas(events: list[SegmentationEvent]) -> list[Sequence]:
    return [event.window for event in events if isinstance(event, LetterEmitted)]


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
    events = run(still_frames(3 + UMBRALES.emit_cooldown_frames))

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

    def counting(sequence: Sequence, origin: WindowOrigin) -> Prediction:  # noqa: ARG001
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
    cooldown = UMBRALES.emit_cooldown_frames
    stream = [*still_frames(3), *moving_frames(cooldown), *still_frames(4)]

    events = run(stream)

    assert emitted(events) == ["A", "A"]


def test_una_ventana_inestable_se_rechaza_antes_de_clasificar() -> None:
    """σ por encima de `quality.max_dispersion`: la ventana no llega al modelo."""
    config = Config.model_validate(
        {
            "quality": {"max_dispersion": 1e-4},
            "segmentation": {
                "buffer_ms": _ms(6),
                "stable_ms": _ms(2),
                "emit_cooldown_ms": _ms(3),
                "reject_cooldown_ms": _ms(2),
                "missing_to_idle_ms": _ms(2),
            },
        }
    )

    def never_called(sequence: Sequence, origin: WindowOrigin) -> Prediction:  # noqa: ARG001
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
    assert all(len(window) <= UMBRALES.buffer_size for window in windows)


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


# --------------------------------------------------------------------------- #
# La ventana clasificada ES el tramo estable (ADR 0004, ADR 0013)
# --------------------------------------------------------------------------- #


def test_la_ventana_clasificada_solo_contiene_frames_estables() -> None:
    """La invariante del ADR 0004, comprobada sobre la ventana que se clasifica.

    «La ventana solo es estable si la mano ni viajó ni siguió acomodándose». La
    implementacion anterior garantizaba eso de los ultimos `stable_run` frames
    —normalmente 5 o 6— y clasificaba los `buffer_size` del buffer circular: con
    un transito mas corto que el buffer, la ventana mezclaba dos manos y el
    promedio del §2 no correspondia a ninguna configuracion real.

    Un viaje de 10 frames seguido de 3 de quietud es exactamente ese caso: el
    buffer que se declara estable todavia contiene cola del transito.
    """
    stream = [*moving_frames(10), *still_frames(3)]

    for window in ventanas(run(stream)):
        assert max(velocidades(window)) < UMBRALES.velocity_threshold


def test_la_ventana_es_el_tramo_estable_y_no_el_buffer_entero() -> None:
    """Se clasifican los frames verificados, no todo lo que se recuerda.

    Con `stable_frames = 2`, dos frames de quietud tras el viaje bastan para
    declarar STABLE, y son exactamente esos dos los que se clasifican: el buffer
    de 6 contiene ademas 4 frames de viaje sobre los que no se comprobo nada.
    """
    stream = [*moving_frames(10), *still_frames(3)]

    emitidas = ventanas(run(stream))

    assert emitidas
    assert len(emitidas[0]) == UMBRALES.stable_frames


def test_la_ventana_sigue_a_stable_run_y_no_a_un_numero_fijo() -> None:
    """El caso mas revelador de la correccion.

    Un rechazo por confianza baja no cambia de estado: la mano sigue quieta y
    `stable_run` sigue creciendo mientras la maquina reintenta. Cuando por fin
    emite lleva 5 pares estables acumulados y la ventana son esos 5 frames — el
    primer frame de la quietud, el que la mano acabo de alcanzar viajando, queda
    fuera. La implementacion anterior clasificaba los 6 del buffer.
    """
    stream = [*moving_frames(10), *still_frames(6)]

    emitidas = ventanas(run(stream, responses(UNSURE_A, CONFIDENT_A)))

    assert emitidas
    assert len(emitidas[0]) == 5
    assert max(velocidades(emitidas[0])) < UMBRALES.velocity_threshold


def test_la_ventana_se_topa_un_frame_por_debajo_del_buffer() -> None:
    """`T_max` es `buffer_size`, pero el tope alcanzable es uno menos.

    Sin transito previo que descartar, la ventana crece con la quietud —
    promediar mas frames reduce mas ruido y aqui todos son de la sena— hasta
    topar. Y topa en `buffer_size - 1`, no en `buffer_size`: `stable_run` cuenta
    PARES de frames, asi que con un buffer lleno de N frames hay N-1 pares y el
    frame mas viejo solo participa como origen del primero. Con `min(stable_run,
    T_max)` ese frame nunca entra en la ventana clasificada.

    No es un error de una unidad: es la lectura conservadora de la definicion, y
    se fija aqui para que `segmentation.ts` no elija la otra en la Fase 7.
    """
    emitidas = ventanas(run(still_frames(20), responses(UNSURE_A, CONFIDENT_A)))

    assert emitidas
    assert len(emitidas[0]) == UMBRALES.buffer_size - 1


# --------------------------------------------------------------------------- #
# Emision progresiva
# --------------------------------------------------------------------------- #


def acumulados(events: list[SegmentationEvent]) -> list[tuple[int, float]]:
    """Longitud de ventana y confianza de cada intento que no emitio."""
    return [
        (len(event.window), event.prediction.confidence)
        for event in events
        if isinstance(event, EvidenceAccumulated)
    ]


def test_la_confianza_alta_emite_de_inmediato() -> None:
    """Latencia adaptativa: una letra sin vecinos cercanos no espera nada.

    En cuanto la ventana es estable y el clasificador la resuelve por encima de
    `high_confidence`, la letra sale con la ventana minima. Es el caso comun —
    medido bajo leave-one-signer-out, el 75% de las muestras.
    """
    eventos = run(still_frames(20), always(CONFIDENT_A))

    emitidas = ventanas(eventos)
    assert len(emitidas) == 1
    assert len(emitidas[0]) == UMBRALES.stable_frames
    assert acumulados(eventos) == []


def test_la_confianza_media_acumula_en_vez_de_emitir() -> None:
    """El corazon del bloque 2.

    Una confianza por encima del piso pero por debajo del umbral alto ya no
    emite en el primer frame estable: sigue clasificando frame a frame mientras
    la ventana crece. La implementacion anterior escribia la letra en cuanto
    superaba `min_confidence`, sin distinguir «segura» de «la menos mala».
    """
    eventos = run(still_frames(20), always(MEDIUM_A))

    assert acumulados(eventos)[0] == (UMBRALES.stable_frames, 0.7)


def test_la_emision_progresiva_emite_en_cuanto_cruza_el_umbral() -> None:
    """Dos intentos medios y uno alto: emite en el tercer frame estable.

    Y lo hace **sin cooldown intermedio**: acumular no es rechazar, asi que la
    maquina reintenta en el frame siguiente y no `reject_cooldown_frames`
    despues. Que los tres indices sean consecutivos es exactamente esa
    propiedad, y es de donde sale la latencia que se recupera.
    """
    eventos = run(still_frames(20), responses(MEDIUM_A, MEDIUM_A, CONFIDENT_A))

    intentos = [
        event.frame_index
        for event in eventos
        if isinstance(event, EvidenceAccumulated | LetterEmitted)
    ]
    primeros = intentos[:3]

    assert primeros == [primeros[0], primeros[0] + 1, primeros[0] + 2]
    assert len(acumulados(eventos)) == 2
    assert acumulados(eventos) == [
        (UMBRALES.stable_frames, 0.7),
        (UMBRALES.stable_frames + 1, 0.7),
    ]
    assert len(ventanas(eventos)[0]) == UMBRALES.stable_frames + 2
    assert RejectionReason.LOW_CONFIDENCE not in rechazos_de(eventos)


def test_agotar_la_ventana_sin_cruzar_el_umbral_emite_igual() -> None:
    """Cuando ya no hay mas evidencia que acumular, la mejor disponible manda.

    Si al agotar la ventana no emitiera, ninguna letra de confianza media
    saldria nunca y el par confundible quedaria mudo en vez de tardar. El piso
    sigue siendo `min_confidence`, que es lo que decidia antes por si solo.
    """
    eventos = run(still_frames(30), always(MEDIUM_A))

    emitidas = ventanas(eventos)
    tope = UMBRALES.buffer_size - 1
    assert emitidas
    assert len(emitidas[0]) == tope
    # La ventana crecio frame a frame desde el piso hasta el tope, y solo el
    # ultimo intento —el que ya no podia crecer mas— emitio.
    assert acumulados(eventos)[: tope - UMBRALES.stable_frames] == [
        (longitud, 0.7) for longitud in range(UMBRALES.stable_frames, tope)
    ]


def test_la_confianza_baja_nunca_emite_ni_acumula() -> None:
    """Por debajo del piso no hay nada que acumular: es un rechazo con su
    cooldown, como antes. Acumular sobre basura solo gastaria clasificaciones."""
    eventos = run(still_frames(20), always(UNSURE_A))

    assert ventanas(eventos) == []
    assert acumulados(eventos) == []
    assert set(rechazos_de(eventos)) == {RejectionReason.LOW_CONFIDENCE}


def rechazos_de(events: list[SegmentationEvent]) -> list[RejectionReason]:
    return [event.reason for event in events if isinstance(event, WindowRejected)]


# --------------------------------------------------------------------------- #
# Los umbrales viven en milisegundos y se derivan de la tasa medida
# --------------------------------------------------------------------------- #


def test_la_conversion_redondea_hacia_arriba_en_el_empate() -> None:
    """La regla de redondeo es parte del contrato, no un detalle.

    `round` de Python redondea al par mas cercano y `Math.round` de JavaScript
    redondea hacia arriba: con 0.5 exacto darian numeros distintos y
    `segmentation.ts` derivaria un cuadro mas o menos que Python con el mismo
    `config.yaml`. Se fija la de JavaScript, `floor(x + 0.5)`.
    """
    # 50 ms a 30 fps son 1.5 cuadros exactos.
    assert frames_from_ms(50.0, 30.0) == 2
    # 150 ms a 30 fps son 4.5 exactos: `round` daria 4, esto da 5.
    assert frames_from_ms(150.0, 30.0) == 5


def test_la_conversion_nunca_baja_de_un_cuadro() -> None:
    """Un cooldown de cero cuadros no es un cooldown: la maquina reclasificaria
    en cada frame, que es justo lo que el cooldown existe para impedir."""
    assert frames_from_ms(1.0, 5.0) == 1
    assert frames_from_ms(0.001, 1.0) == 1


def test_a_treinta_fps_los_umbrales_dan_los_valores_historicos() -> None:
    """Los milisegundos por defecto son los que los comentarios de `config.yaml`
    ya afirmaban: 24 cuadros de buffer, 5 de estabilidad, 12 de cooldown. Lo que
    cambia no es la intencion, es que ahora se cumple a cualquier tasa."""
    umbrales = FrameThresholds.from_config(Config(), fps=30.0)

    assert umbrales.buffer_size == 24
    assert umbrales.stable_frames == 5
    assert umbrales.emit_cooldown_frames == 12
    assert umbrales.reject_cooldown_frames == 4
    assert umbrales.missing_frames_to_idle == 8


def test_a_la_tasa_medida_de_verdad_los_umbrales_encogen() -> None:
    """El defecto que este bloque repara, en un solo test.

    A los 17.8 fps medidos en la maquina de referencia, los mismos 800 ms de
    buffer son 14 cuadros y no 24. Antes el numero de cuadros era fijo y lo que
    variaba era el tiempo: 24 cuadros eran 800 ms en el papel y 1348 ms en la
    maquina. Ahora es al reves, que es lo que quien firma percibe.
    """
    umbrales = FrameThresholds.from_config(Config(), fps=17.8)

    assert umbrales.buffer_size == 14
    assert umbrales.stable_frames == 3
    assert umbrales.emit_cooldown_frames == 7


def test_la_maquina_usa_la_tasa_que_se_le_pasa() -> None:
    """Y no una constante escondida: a la mitad de tasa, la misma quietud fisica
    exige la mitad de cuadros para declararse estable."""
    config = Config.model_validate({"segmentation": {"stable_ms": 200.0}})

    rapida = list(
        run_segmentation(still_frames(30), config, always(CONFIDENT_A), fps=30.0)
    )
    lenta = list(
        run_segmentation(still_frames(30), config, always(CONFIDENT_A), fps=15.0)
    )

    def primera_emision(events: list[SegmentationEvent]) -> int:
        return next(e.frame_index for e in events if isinstance(e, LetterEmitted))

    assert primera_emision(rapida) == 6  # 200 ms a 30 fps son 6 cuadros
    assert primera_emision(lenta) == 3  # y a 15 fps, 3


# --------------------------------------------------------------------------- #
# El camino dinámico (§6.7, ADR 0015)
# --------------------------------------------------------------------------- #
#
# La mano sintética mide ~100 px de muñeca a nudillo, así que un paso de k px por
# frame da `v_t ≈ k / 100`: 10 px es trazo franco (0.10), 2.2 px cae entre
# `velocity_threshold` y `motion_threshold` (0.022) y 0 px es quietud.

#: Umbrales apretados del camino dinámico, a 30 fps. Los del camino estático son
#: los de `CONFIG`; lo que cambia es que aquí un trazo de 4 frames móviles ya es
#: candidato y 4 frames de reposo lo cierran.
DYNAMIC_CONFIG = Config.model_validate(
    {
        "plausibility": {"enabled": False},
        "segmentation": {
            **CONFIG.segmentation.model_dump(),
            "velocity_threshold_per_s": 0.02 * FPS,
            # Un cuadro: la velocidad del §6.1 entre pares consecutivos. Estos
            # tests son de la lógica de la máquina; la ventana tiene los suyos.
            "velocity_window_ms": _ms(1),
            "closing_window_ms": _ms(1),
            "motion_threshold_per_s": 0.025 * FPS,
            "motion_min_ms": _ms(4),
            "motion_confirm_low_ms": _ms(4),
            "motion_max_ms": _ms(30),
        },
    }
)
DYNAMIC_UMBRALES = FrameThresholds.from_config(DYNAMIC_CONFIG, FPS)

CONFIDENT_J = Prediction(label="J", confidence=0.93)


def trazo(
    *segments: tuple[float, float, int], start: tuple[float, float] = (400.0, 300.0)
) -> list[RawFrame]:
    """Un recorrido por tramos `(dx, dy, frames)`, en píxeles por frame.

    El primer frame está en `start`; cada tramo añade `frames` frames movidos
    `(dx, dy)` desde el anterior. Un tramo `(0, 0, n)` es una pausa: la mano
    frena en un cambio de dirección.
    """
    x, y = start
    frames = [to_frame(translated(canonical_hand(), x, y), width=1280, height=720)]
    for dx, dy, count in segments:
        for _ in range(count):
            x += dx
            y += dy
            frames.append(
                to_frame(translated(canonical_hand(), x, y), width=1280, height=720)
            )
    return frames


def quieta_tras(frames: list[RawFrame], count: int) -> list[RawFrame]:
    """La mano detenida donde terminó el trazo, `count` frames."""
    return [frames[-1]] * count


def muneca_px(frames: list[RawFrame]) -> tuple[float, float]:
    """Dónde quedó la muñeca, en píxeles, en el último frame de un trazo."""
    ultimo = frames[-1]
    return (ultimo.landmarks[0].x * 1280, ultimo.landmarks[0].y * 720)


class PorOrigen:
    """Doble de clasificador que responde según la ruta y anota quién le llamó."""

    def __init__(
        self,
        stable: Prediction = CONFIDENT_A,
        dynamic: Prediction = CONFIDENT_J,
    ) -> None:
        self.responses = {WindowOrigin.STABLE: stable, WindowOrigin.DYNAMIC: dynamic}
        self.calls: list[tuple[WindowOrigin, Sequence]] = []

    def __call__(self, sequence: Sequence, origin: WindowOrigin) -> Prediction:
        self.calls.append((origin, sequence))
        return self.responses[origin]

    def origins(self) -> list[WindowOrigin]:
        return [origin for origin, _ in self.calls]


def run_dynamic(
    stream: Iterable[FrameSlot], classify: Classify, config: Config = DYNAMIC_CONFIG
) -> list[SegmentationEvent]:
    return list(run_segmentation(stream, config, classify, fps=FPS))


def dynamic_windows(events: list[SegmentationEvent]) -> list[Sequence]:
    return [e.window for e in events if isinstance(e, WindowDynamic)]


def reached(events: list[SegmentationEvent]) -> set[State]:
    return {current for _, current in transitions(events)}


def test_una_dinamica_limpia_se_entrega_entera() -> None:
    """El trazo completo, del primer frame al último en movimiento, y nada más.

    Ni el reposo que lo cierra —no es parte de la letra— ni un recorte del
    principio: la trayectoria entera es lo que distingue la letra, y el §3.1 mide
    τ desde el primer frame.
    """
    stroke = trazo((10.0, 0.0, 6), (0.0, 10.0, 6))
    clasificador = PorOrigen()

    events = run_dynamic([*stroke, *quieta_tras(stroke, 8)], clasificador)

    assert dynamic_windows(events) == [Sequence(frames=tuple(stroke))]
    assert clasificador.origins() == [WindowOrigin.DYNAMIC]
    assert emitted(events) == ["J"]
    emision = next(e for e in events if isinstance(e, LetterEmitted))
    assert emision.origin is WindowOrigin.DYNAMIC
    assert (State.TRACKING, State.DYNAMIC_CANDIDATE) in transitions(events)
    assert (State.DYNAMIC_CANDIDATE, State.DYNAMIC_EMIT) in transitions(events)
    assert (State.DYNAMIC_EMIT, State.EMIT) in transitions(events)


def test_la_pausa_de_un_cambio_de_direccion_no_parte_la_dinamica() -> None:
    """La Z: tres tramos con dos frenos, cada uno más largo que `stable_frames`.

    Cada freno deja la velocidad en cero tres frames, y `stable_frames` son dos:
    el camino estático, por su cuenta, declararía STABLE en mitad del trazo y
    clasificaría el freno como letra. Con el candidato abierto, STABLE está
    suspendido y los tres tramos llegan juntos al clasificador dinámico.
    """
    pausa = DYNAMIC_UMBRALES.stable_frames + 1
    assert pausa < DYNAMIC_UMBRALES.motion_confirm_low_frames
    stroke = trazo(
        (10.0, 0.0, 6),
        (0.0, 0.0, pausa),
        (-8.0, 8.0, 6),
        (0.0, 0.0, pausa),
        (10.0, 0.0, 6),
    )
    clasificador = PorOrigen()

    events = run_dynamic([*stroke, *quieta_tras(stroke, 8)], clasificador)

    assert dynamic_windows(events) == [Sequence(frames=tuple(stroke))]
    assert WindowOrigin.STABLE not in clasificador.origins()
    assert not any(isinstance(e, WindowStable) for e in events)
    assert emitted(events) == ["J"]


def test_sin_la_suspension_el_freno_si_se_leeria_como_estatica() -> None:
    """El contraste del test anterior: el defecto que la suspensión corrige.

    Con el camino dinámico fuera de alcance —un `motion_min` que ningún trazo
    alcanza—, la misma Z pasa por STABLE en su primer freno. Si este test dejara
    de cumplirse, el anterior ya no estaría probando nada.
    """
    sin_camino = Config.model_validate(
        {
            "segmentation": {
                **DYNAMIC_CONFIG.segmentation.model_dump(),
                "motion_min_ms": _ms(200),
                "motion_max_ms": _ms(400),
            }
        }
    )
    pausa = DYNAMIC_UMBRALES.stable_frames + 1
    stroke = trazo((10.0, 0.0, 6), (0.0, 0.0, pausa), (-8.0, 8.0, 6))
    clasificador = PorOrigen()

    events = run_dynamic(stroke, clasificador, sin_camino)

    assert WindowOrigin.STABLE in clasificador.origins()
    assert (State.TRACKING, State.STABLE) in transitions(events)


def test_un_transito_que_no_llega_a_motion_min_no_se_vuelve_candidato() -> None:
    """Movimiento franco pero corto: el viaje de una letra a la siguiente.

    Cada frame supera `motion_threshold` y aun así la racha no alcanza
    `motion_min`: no hay candidato, el clasificador dinámico no se entera, y la
    letra de llegada sale por el camino estático como siempre. Nada se emite a
    medias.
    """
    llegada = trazo((10.0, 0.0, DYNAMIC_UMBRALES.motion_min_frames - 1))
    stream = [*quieta_tras(llegada[:1], 6), *llegada, *quieta_tras(llegada, 6)]
    clasificador = PorOrigen()

    events = run_dynamic(stream, clasificador)

    assert State.DYNAMIC_CANDIDATE not in reached(events)
    assert dynamic_windows(events) == []
    assert WindowOrigin.DYNAMIC not in clasificador.origins()
    assert emitted(events) == ["A", "A"]


def test_un_arrastre_bajo_motion_threshold_no_cuenta_como_trazo() -> None:
    """La otra forma de casi cruzar: lento pero largo.

    A 2.2 px por frame la mano no está quieta para el camino estático
    (`v_t` ≈ 0.022 > `velocity_threshold`) pero tampoco se mueve para el dinámico
    (< `motion_threshold`). Por largo que sea, no hay racha que acumular.
    """
    arrastre = trazo((2.2, 0.0, 4 * DYNAMIC_UMBRALES.motion_max_frames))
    clasificador = PorOrigen()

    events = run_dynamic(arrastre, clasificador)

    assert State.DYNAMIC_CANDIDATE not in reached(events)
    assert clasificador.calls == []


def test_una_estatica_pura_nunca_entra_a_candidato() -> None:
    """Mano quieta con temblor de los dedos: solo existe el camino estático."""
    clasificador = PorOrigen()

    events = run_dynamic([*still_frames(10), *jittery_frames(40)], clasificador)

    assert State.DYNAMIC_CANDIDATE not in reached(events)
    assert set(clasificador.origins()) == {WindowOrigin.STABLE}


def test_los_transitos_cortos_de_un_deletreo_rapido_no_se_suman() -> None:
    """Tres viajes cortos separados por paradas de letra estática.

    Juntos superan `motion_min`, pero cada parada dura lo que el camino estático
    llama quietud y corta la racha. Sin ese corte, a la tercera letra el
    deletreo entero parecería un trazo largo.

    El estático duda en cada parada a propósito: si emitiera, el viaje siguiente
    caería dentro del cooldown de EMIT, donde no se entra a candidato, y el test
    pasaría sin haber puesto a prueba el corte.
    """
    corto = DYNAMIC_UMBRALES.motion_min_frames - 2
    parada = DYNAMIC_UMBRALES.stable_frames + 1
    assert parada < DYNAMIC_UMBRALES.motion_confirm_low_frames
    assert 3 * (corto + 1) >= DYNAMIC_UMBRALES.motion_min_frames
    stroke = trazo(
        (10.0, 0.0, corto),
        (0.0, 0.0, parada),
        (0.0, 10.0, corto),
        (0.0, 0.0, parada),
        (-10.0, 0.0, corto),
    )
    clasificador = PorOrigen(stable=UNSURE_A)

    events = run_dynamic([*stroke, *quieta_tras(stroke, 6)], clasificador)

    assert State.DYNAMIC_CANDIDATE not in reached(events)
    assert WindowOrigin.DYNAMIC not in clasificador.origins()


def test_un_transito_largo_rechazado_deja_paso_a_la_estatica() -> None:
    """Un viaje largo sí se vuelve candidato; el DTW lo rechaza y la letra de
    llegada sale en el frame siguiente, sin volver a esperar `stable_frames`:
    el reposo que cerró el candidato ya contaba como quietud."""
    viaje = trazo((10.0, 0.0, DYNAMIC_UMBRALES.motion_min_frames + 3))
    clasificador = PorOrigen(dynamic=Prediction.unknown(confidence=0.55))

    events = run_dynamic([*viaje, *quieta_tras(viaje, 10)], clasificador)

    rechazo = next(e for e in events if isinstance(e, WindowRejected))
    assert rechazo.reason is RejectionReason.LOW_CONFIDENCE
    assert (State.DYNAMIC_EMIT, State.TRACKING) in transitions(events)
    assert emitted(events) == ["A"]
    emision = next(e for e in events if isinstance(e, LetterEmitted))
    assert emision.frame_index == rechazo.frame_index + 1
    assert emision.origin is WindowOrigin.STABLE


def test_la_pose_final_de_una_dinamica_no_se_emite_como_estatica() -> None:
    """La J termina en la mano de la I. Sostenerla después del trazo no escribe
    una I: hasta que la mano se mueva, el camino estático no promueve a STABLE.
    El movimiento siguiente libera el cerrojo, como el rebote de las dobles."""
    stroke = trazo((10.0, 0.0, 6), (0.0, 10.0, 6))
    sostenida = quieta_tras(stroke, 40)
    rebote = trazo((10.0, 0.0, 2), start=muneca_px(stroke))
    clasificador = PorOrigen()

    events = run_dynamic(
        [*stroke, *sostenida, *rebote, *quieta_tras(rebote, 8)], clasificador
    )

    assert emitted(events) == ["J", "A"]
    primera_estatica = clasificador.origins().index(WindowOrigin.STABLE)
    assert clasificador.origins()[:primera_estatica] == [WindowOrigin.DYNAMIC]


def test_un_candidato_demasiado_largo_se_descarta_sin_clasificar() -> None:
    """Movimiento que no para: se descarta al pasar `motion_max` sin clasificar.

    Tras el rechazo no nace otro candidato durante `motion_exhausted_ms`
    (v6, ADR 0019): alguien gesticulando sin parar produce como mucho un rechazo
    por cada `motion_max + motion_exhausted`, y ninguno llega al clasificador.
    Hasta la v5 no nacía ninguno más mientras la mano no reposara, y en la X y
    la Q —que nunca reposan del todo— eso bloqueaba los intentos siguientes.
    """
    total = 3 * DYNAMIC_UMBRALES.motion_max_frames
    agitada = trazo((10.0, 0.0, total))
    clasificador = PorOrigen()

    events = run_dynamic(agitada, clasificador)

    rechazos = [e for e in events if isinstance(e, WindowRejected)]
    assert {e.reason for e in rechazos} == {RejectionReason.DYNAMIC_TOO_LONG}
    ciclo = (
        DYNAMIC_UMBRALES.motion_max_frames + DYNAMIC_UMBRALES.motion_exhausted_frames
    )
    assert 1 <= len(rechazos) <= -(-total // ciclo)
    assert clasificador.calls == []
    entradas = [
        e.frame_index
        for e in events
        if isinstance(e, StateChanged) and e.current is State.DYNAMIC_CANDIDATE
    ]
    for rechazo, siguiente in zip(rechazos, entradas[1:], strict=False):
        assert (
            siguiente - rechazo.frame_index >= DYNAMIC_UMBRALES.motion_exhausted_frames
        )


def test_tras_un_demasiado_largo_la_espera_fija_deja_nacer_otro_trazo() -> None:
    """La tercera salida de la cascada (v6): sin reposo y sin hueco, a los
    `motion_exhausted_ms` del rechazo la mano puede volver a empezar un trazo."""
    agitada = trazo(
        (10.0, 0.0, DYNAMIC_UMBRALES.motion_max_frames + 2),
        (
            0.0,
            10.0,
            DYNAMIC_UMBRALES.motion_exhausted_frames
            + 3 * DYNAMIC_UMBRALES.motion_min_frames,
        ),
    )

    events = run_dynamic(agitada, PorOrigen())

    entradas = [
        e
        for e in events
        if isinstance(e, StateChanged) and e.current is State.DYNAMIC_CANDIDATE
    ]
    assert len(entradas) == 2


def test_un_hueco_largo_descarta_el_trazo_entero() -> None:
    """Sin coser (§0.3): un hueco más largo que `dynamic_max_gap_ms` en mitad del
    trazo lo invalida, y lo que viene después empieza de cero. Los cortos se
    rellenan desde el Bloque 2 (siguientes tests)."""
    antes = trazo((10.0, 0.0, 8))
    despues = trazo((0.0, 10.0, 2), start=muneca_px(antes))
    clasificador = PorOrigen()
    largo = DYNAMIC_UMBRALES_HUECOS.max_gap_frames + 1

    events = run_dynamic(
        [*antes, *missing_frames(largo), *despues, *quieta_tras(despues, 8)],
        clasificador,
    )

    rechazos = [e.reason for e in events if isinstance(e, WindowRejected)]
    assert RejectionReason.DYNAMIC_INTERRUPTED in rechazos
    assert dynamic_windows(events) == []
    assert WindowOrigin.DYNAMIC not in clasificador.origins()


def test_los_umbrales_del_camino_dinamico_se_convierten_a_cuadros() -> None:
    """Como el resto: milisegundos en `config.yaml`, cuadros con la tasa."""
    umbrales = FrameThresholds.from_config(Config(), fps=30.0)

    assert umbrales.motion_min_frames == 10
    assert umbrales.motion_confirm_low_frames == 20  # 667 ms, v5 provisional
    assert umbrales.motion_max_frames == 120


def test_el_cierre_mira_su_propia_ventana_y_el_resto_la_suya() -> None:
    """v6 (§6.1.2): con `closing_window_ms` más larga que `velocity_window_ms`, el
    cierre de un trazo que para en seco llega `closing - velocity` cuadros más
    tarde; sin candidato, nada cambia."""
    corta = Config.model_validate(
        {
            "segmentation": {
                **DYNAMIC_CONFIG.segmentation.model_dump(),
                "closing_window_ms": _ms(1),
            }
        }
    )
    larga = Config.model_validate(
        {
            "segmentation": {
                **DYNAMIC_CONFIG.segmentation.model_dump(),
                "velocity_window_ms": _ms(1),
                "closing_window_ms": _ms(3),
            }
        }
    )
    frames = trazo((10.0, 0.0, 8))
    flujo = [*frames, *quieta_tras(frames, 10)]

    def cierre(config: Config) -> int:
        eventos = list(run_segmentation(iter(flujo), config, PorOrigen(), fps=FPS))
        (trazo_entregado,) = [e for e in eventos if isinstance(e, WindowDynamic)]
        return trazo_entregado.frame_index

    assert cierre(larga) - cierre(corta) == 2


# --------------------------------------------------------------------------- #
# Bloque 2: huecos cortos dentro de un trazo
# --------------------------------------------------------------------------- #

DYNAMIC_UMBRALES_HUECOS = GapPolicy.from_config(DYNAMIC_CONFIG, FPS)


def _trazo_con_hueco(hueco: int, largo: int = 12) -> list[FrameSlot]:
    """Un trazo de `largo` frames en el que se pierden `hueco` a partir del octavo.

    Los perdidos **reemplazan** frames: la mano sigue moviéndose durante el
    hueco, como en vivo. Si se insertaran, la interpolación entre dos frames
    contiguos parecería reposo y cerraría el trazo a mitad del hueco.
    """
    frames = trazo((10.0, 0.0, largo + hueco))
    return [
        *frames[:8],
        *missing_frames(hueco),
        *frames[8 + hueco :],
        *quieta_tras(frames, 8),
    ]


def test_un_hueco_corto_en_el_trazo_se_rellena_y_el_trazo_sale_entero() -> None:
    clasificador = PorOrigen()

    events = run_dynamic(_trazo_con_hueco(2), clasificador)

    (entregado,) = [e for e in events if isinstance(e, WindowDynamic)]
    assert entregado.interpolated_frames == 2
    assert RejectionReason.DYNAMIC_INTERRUPTED not in [
        e.reason for e in events if isinstance(e, WindowRejected)
    ]
    # Los interpolados caen sobre la recta entre los dos extremos.
    x = [f.landmarks[0].x for f in entregado.window.frames]
    assert x == sorted(x)


def test_el_hueco_mas_largo_que_se_rellena_es_max_gap() -> None:
    """Con un trazo lo bastante largo para que la fracción interpolada no
    decida (5 de 23 < 25%) ni el largo máximo: lo que separa los dos casos es el
    hueco."""
    justo = DYNAMIC_UMBRALES_HUECOS.max_gap_frames

    rellenado = run_dynamic(_trazo_con_hueco(justo, largo=18), PorOrigen())
    cortado = run_dynamic(_trazo_con_hueco(justo + 1, largo=18), PorOrigen())

    assert len([e for e in rellenado if isinstance(e, WindowDynamic)]) == 1
    assert RejectionReason.DYNAMIC_INTERRUPTED in [
        e.reason for e in cortado if isinstance(e, WindowRejected)
    ]


def test_un_hueco_con_cambio_de_mano_no_se_rellena() -> None:
    """La mano declarada cambia a través del hueco: no es la misma mano."""
    frames = trazo((10.0, 0.0, 12))
    izquierda = [replace(f, handedness=Handedness.LEFT) for f in frames[8:]]
    flujo = [*frames[:8], *missing_frames(1), *izquierda, *quieta_tras(izquierda, 8)]

    events = run_dynamic(flujo, PorOrigen())

    assert RejectionReason.DYNAMIC_INTERRUPTED in [
        e.reason for e in events if isinstance(e, WindowRejected)
    ]


def test_un_trazo_demasiado_reconstruido_se_descarta_sin_clasificar() -> None:
    config = Config.model_validate(
        {
            "segmentation": {
                **DYNAMIC_CONFIG.segmentation.model_dump(),
                "dynamic_max_interpolated_fraction": 0.05,
            }
        }
    )
    clasificador = PorOrigen()

    events = list(
        run_segmentation(iter(_trazo_con_hueco(2)), config, clasificador, fps=FPS)
    )

    assert RejectionReason.DYNAMIC_TOO_MUCH_INTERPOLATED in [
        e.reason for e in events if isinstance(e, WindowRejected)
    ]
    assert [e for e in events if isinstance(e, WindowDynamic)] == []
    # Tras el rechazo el camino estático toma la mano quieta, como siempre;
    # lo que no ocurre es que el trazo llegue al clasificador dinámico.
    assert [c for c in clasificador.calls if c[0] is WindowOrigin.DYNAMIC] == []


def test_fuera_del_candidato_un_hueco_sigue_interrumpiendo() -> None:
    """El camino estático no cambia (Bloque 2, punto 9): con la mano quieta, un
    frame perdido vacía el buffer como siempre."""
    quieta = quieta_tras(trazo((10.0, 0.0, 1)), 20)
    flujo = [*quieta[:10], *missing_frames(1), *quieta[10:]]

    events = run_dynamic(flujo, PorOrigen())

    assert [e for e in events if isinstance(e, WindowDynamic)] == []
    estables = [
        e for e in events if isinstance(e, StateChanged) and e.current is State.STABLE
    ]
    # Tras el hueco la ventana estable tiene que volver a llenarse desde cero.
    assert len(estables) >= 1


# --------------------------------------------------------------------------- #
# Paso 4 (ADR 0029): el hueco se sostiene también en TRACKING y en STABLE
# --------------------------------------------------------------------------- #

#: La máquina del camino dinámico con el sostén fuera del trazo: dos cuadros en
#: TRACKING y uno en STABLE.
SOSTEN_CONFIG = Config.model_validate(
    {
        "plausibility": {"enabled": False},
        "segmentation": {
            **DYNAMIC_CONFIG.segmentation.model_dump(),
            "tracking_max_gap_ms": _ms(2),
            "stable_max_gap_ms": _ms(1),
        },
    }
)


def _huecos(events: list[SegmentationEvent]) -> list[GapResolved]:
    return [e for e in events if isinstance(e, GapResolved)]


def test_un_parpadeo_en_stable_no_reinicia_la_ventana() -> None:
    """Un cuadro sin mano con la ventana estable: se sostiene, se rellena y la
    máquina sigue en STABLE en vez de volver a TRACKING y esperar otra vez."""
    events = run_dynamic(
        [*still_frames(6), *missing_frames(1), *still_frames(4)],
        PorOrigen(stable=UNSURE_A),
        SOSTEN_CONFIG,
    )

    assert _huecos(events) == [
        GapResolved(
            frame_index=7, state=State.STABLE, frames=1, implausible=0, filled=True
        )
    ]
    assert (State.STABLE, State.TRACKING) not in transitions(events)
    assert not any(isinstance(e, HandLost) for e in events)


def test_un_hueco_mas_largo_que_el_limite_de_stable_se_procesa_como_siempre() -> None:
    events = run_dynamic(
        [*still_frames(6), *missing_frames(2), *still_frames(4)],
        PorOrigen(stable=UNSURE_A),
        SOSTEN_CONFIG,
    )

    huecos = _huecos(events)
    assert [(h.state, h.frames, h.filled) for h in huecos] == [(State.STABLE, 2, False)]
    assert (State.STABLE, State.IDLE) in transitions(events)


def test_perder_la_mano_al_arrancar_el_trazo_ya_no_lo_mata() -> None:
    """En TRACKING, antes de llegar a candidato: el trazo se sostiene y se
    entrega entero, con los rellenados contados."""
    frames = trazo((10.0, 0.0, 12))
    flujo = [*frames[:3], *missing_frames(2), *frames[5:], *quieta_tras(frames, 8)]
    clasificador = PorOrigen()

    con_sosten = run_dynamic(flujo, clasificador, SOSTEN_CONFIG)
    sin_sosten = run_dynamic(flujo, PorOrigen())

    trazos = [e for e in con_sosten if isinstance(e, WindowDynamic)]
    assert len(trazos) == 1
    assert trazos[0].interpolated_frames == 2
    assert [h.state for h in _huecos(con_sosten)] == [State.TRACKING]
    assert [e for e in sin_sosten if isinstance(e, WindowDynamic)] == [] or all(
        len(e.window) < len(trazos[0].window)
        for e in sin_sosten
        if isinstance(e, WindowDynamic)
    )
