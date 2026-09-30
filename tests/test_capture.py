"""El criterio con el que se acepta una muestra, probado sin cámara.

Es la mitad del proyecto que decide qué acaba en el dataset, y lo que decide mal
aquí no se arregla después: una muestra temblorosa aceptada por error es ruido
etiquetado que nadie va a volver a mirar, y aparecerá en la Fase 2 como una clase
que no separa.

Todo lo de este archivo corre en una máquina sin webcam, sin MediaPipe y sin
OpenCV. Esa es la razón de que `lsm/capture.py` exista separado del CLI.
"""

from __future__ import annotations

import pytest

from lsm.capture import (
    BufferedFrame,
    FrameBuffer,
    Rejection,
    evaluate_window,
    explain,
    maximum_frames,
    minimum_frames,
    preview_position,
    trajectory_arc_length,
)
from lsm.config import Config
from lsm.features import ExtractionRejected, extract_sequence_features
from lsm.synthetic import (
    arc_offsets,
    canonical_hand,
    collapsed_scale,
    fist_hand,
    moving_sequence,
    still_sequence,
    to_frame,
    translated,
)
from lsm.types import (
    FrameSlot,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Landmark,
    RawFrame,
    SampleKind,
    Sequence,
)

LUZ = 0.42


def bufferizar(sequence: Sequence, luminance: float = LUZ) -> tuple[BufferedFrame, ...]:
    return tuple(
        BufferedFrame(slot=frame, luminance=luminance) for frame in sequence.frames
    )


def quieta(config: Config) -> tuple[BufferedFrame, ...]:
    """Una estática impecable: la misma mano repetida, σ nula."""
    return bufferizar(
        still_sequence(canonical_hand(), length=config.capture.static_frames)
    )


def cambiando_de_forma(length: int) -> tuple[BufferedFrame, ...]:
    """Una mano que se cierra a lo largo de la ventana: σ alta de verdad.

    **Tiene que cambiar la configuración, no la posición.** σ se calcula sobre el
    vector del §2, que ya pasó por la traslación del paso 3 y la escala del paso 4:
    es invariante a dónde está la mano y a cuánto se acerca. Una ventana en la que
    la mano viaja por el encuadre sin cambiar de forma tiene σ ≈ 0 y se acepta con
    razón — la seña es la misma. Lo verifica
    `test_la_dispersion_no_mide_el_viaje_de_la_mano_sino_su_forma`.
    """
    abierta, cerrada = canonical_hand(), fist_hand()
    frames = []
    for indice in range(length):
        t = indice / (length - 1)
        mezcla = tuple(
            (
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * (b[2] - a[2]),
            )
            for a, b in zip(abierta, cerrada, strict=True)
        )
        frames.append(
            BufferedFrame(slot=to_frame(mezcla, width=1280, height=720), luminance=LUZ)
        )
    return tuple(frames)


def dinamica_realista(length: int) -> tuple[BufferedFrame, ...]:
    """Una J creible: la mano cambia de forma **y** recorre un trazo.

    Es la unica combinacion que separa los dos criterios opuestos. Una mano que
    solo cambia de forma no viaja y no tiene arco; una que solo viaja no cambia de
    forma y tiene sigma nula. Una sena dinamica de verdad hace las dos cosas, asi
    que es la que debe caer del lado del rechazo estatico y del lado de la
    aceptacion dinamica.
    """
    abierta, cerrada = canonical_hand(), fist_hand()
    offsets = arc_offsets(length, width_px=140.0, depth_px=180.0)
    frames = []
    for indice, (dx, dy) in enumerate(offsets):
        t = indice / (length - 1)
        mezcla = tuple(
            (
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * (b[2] - a[2]),
            )
            for a, b in zip(abierta, cerrada, strict=True)
        )
        frames.append(
            BufferedFrame(
                slot=to_frame(translated(mezcla, dx, dy), width=1280, height=720),
                luminance=LUZ,
            )
        )
    return tuple(frames)


def arco_de(frames: tuple[BufferedFrame, ...], config: Config) -> float:
    """La longitud de arco de una ventana, para contrastarla con el umbral."""
    slots = tuple(buffered.slot for buffered in frames)
    secuencia = Sequence(frames=slots)  # type: ignore[arg-type]
    extraccion = extract_sequence_features(secuencia, config)
    assert not isinstance(extraccion, ExtractionRejected)
    return trajectory_arc_length(extraccion.trajectory)


# --------------------------------------------------------------------------- #
# El buffer circular
# --------------------------------------------------------------------------- #


def test_el_buffer_conserva_los_ultimos_frames_y_olvida_los_viejos() -> None:
    buffer = FrameBuffer(capacity=3)
    for indice in range(5):
        buffer.push(InvalidFrame(reason=InvalidReason.NO_HAND, detail=str(indice)), LUZ)

    assert len(buffer) == 3
    detalles = [
        slot.detail
        for slot in (frame.slot for frame in buffer.tail(3))
        if isinstance(slot, InvalidFrame)
    ]
    assert detalles == ["2", "3", "4"]


def test_pedirle_al_buffer_mas_de_lo_que_tiene_devuelve_lo_que_hay() -> None:
    """El primer segundo de la sesión: el buffer todavía se está llenando y el
    preview tiene que seguir dibujando."""
    buffer = FrameBuffer(capacity=10)
    buffer.push(InvalidFrame(reason=InvalidReason.NO_HAND), LUZ)

    assert len(buffer.tail(10)) == 1
    assert buffer.tail(0) == ()


def test_una_capacidad_no_positiva_se_rechaza() -> None:
    with pytest.raises(ValueError, match="capacidad"):
        FrameBuffer(capacity=0)


def test_una_luminancia_fuera_de_rango_se_rechaza() -> None:
    """`Sample` la exige en [0, 1] y se valida donde entra, no donde revienta."""
    with pytest.raises(ValueError, match="luminancia"):
        BufferedFrame(slot=InvalidFrame(reason=InvalidReason.NO_HAND), luminance=1.5)


# --------------------------------------------------------------------------- #
# Aceptación de la ventana
# --------------------------------------------------------------------------- #


def test_una_estatica_quieta_se_acepta() -> None:
    config = Config()

    quality = evaluate_window(quieta(config), SampleKind.STATIC, config)

    assert quality.accepted
    assert quality.rejection is None
    # No es cero exacto: el §2 resta la media antes de elevar al cuadrado, y sobre
    # frames idénticos eso deja un residuo del orden de 1e-16. Fijar `== 0.0`
    # ataría el test a la aritmética de una máquina concreta.
    assert quality.dispersion == pytest.approx(0.0, abs=1e-12)
    assert quality.handedness is Handedness.RIGHT
    assert quality.mean_luminance == pytest.approx(LUZ)
    assert quality.mean_scale_px is not None
    assert quality.mean_scale_px > 0.0


def test_una_ventana_corta_se_rechaza_por_frames_y_no_por_otra_cosa() -> None:
    config = Config()
    frames = quieta(config)[: config.capture.static_frames - 1]

    quality = evaluate_window(frames, SampleKind.STATIC, config)

    assert quality.rejection is Rejection.TOO_FEW_FRAMES
    assert quality.dispersion is None


def _con_hueco(
    frames: tuple[BufferedFrame, ...], largo: int
) -> tuple[BufferedFrame, ...]:
    hueco = BufferedFrame(
        slot=InvalidFrame(reason=InvalidReason.NO_HAND), luminance=LUZ
    )
    return (*frames[:5], *(hueco,) * largo, *frames[5:])


def test_un_hueco_largo_en_la_ventana_la_rechaza() -> None:
    """Si el detector perdió la mano más de `stable_max_gap_ms`, la muestra no
    describe una seña continua y coserla inventaría lo que nadie ejecutó."""
    config = Config()

    quality = evaluate_window(_con_hueco(quieta(config), 4), SampleKind.STATIC, config)

    assert quality.rejection is Rejection.HAS_GAPS


def test_un_parpadeo_del_detector_en_una_estatica_se_rellena() -> None:
    """Paso 4 (ADR 0029): un hueco de hasta `stable_max_gap_ms` en una estática
    se rellena, como en vivo en STABLE, y la muestra se acepta."""
    config = Config()

    quality = evaluate_window(_con_hueco(quieta(config), 1), SampleKind.STATIC, config)

    assert quality.accepted


def test_sin_relleno_estatico_cualquier_hueco_la_rechaza() -> None:
    """Con `stable_max_gap_ms: 0` (la regla de antes del Paso 4)."""
    config = Config.model_validate({"segmentation": {"stable_max_gap_ms": 0.0}})

    quality = evaluate_window(_con_hueco(quieta(config), 1), SampleKind.STATIC, config)

    assert quality.rejection is Rejection.HAS_GAPS


def test_un_hueco_al_final_se_recorta() -> None:
    """Los huecos del borde no se rellenan: se recortan, y la muestra queda con
    los frames válidos."""
    config = Config()
    frames = (
        *quieta(config),
        BufferedFrame(slot=InvalidFrame(reason=InvalidReason.NO_HAND), luminance=LUZ),
    )

    quality = evaluate_window(frames, SampleKind.STATIC, config)

    assert quality.accepted


def test_una_estatica_temblorosa_se_rechaza_por_sigma() -> None:
    config = Config()

    quality = evaluate_window(
        cambiando_de_forma(config.capture.static_frames), SampleKind.STATIC, config
    )

    assert quality.rejection is Rejection.UNSTABLE


def test_la_sigma_se_informa_aunque_la_ventana_se_rechace() -> None:
    """Es lo que hace útil la barra del preview: quien graba tiene que ver
    *cuánto* se pasa del umbral, no solo que se pasó."""
    config = Config()

    quality = evaluate_window(
        cambiando_de_forma(config.capture.static_frames), SampleKind.STATIC, config
    )

    assert quality.dispersion is not None
    assert quality.dispersion > config.quality.max_dispersion


def test_una_dinamica_se_rechaza_como_estatica_y_se_acepta_como_dinamica() -> None:
    """La asimetría que define los dos modos.

    En una J, una Ñ o una Z el movimiento **es** la seña (`feature-spec.md` §6.3).
    Aplicarles el umbral de quietud rechazaría exactamente las muestras correctas,
    y el dataset se quedaría sin las ocho letras con recorrido.
    """
    config = Config()
    frames = dinamica_realista(config.capture.static_frames)

    estatica = evaluate_window(frames, SampleKind.STATIC, config)
    dinamica = evaluate_window(frames, SampleKind.DYNAMIC, config)

    assert estatica.rejection is Rejection.UNSTABLE
    assert dinamica.accepted
    # Los dos números se calculan siempre; lo que cambia es cuál decide.
    assert dinamica.dispersion == estatica.dispersion
    assert dinamica.arc_length == estatica.arc_length


# --------------------------------------------------------------------------- #
# El criterio de las dinámicas: longitud de arco
# --------------------------------------------------------------------------- #


def test_el_arco_de_una_mano_quieta_es_casi_cero() -> None:
    """El suelo de la medida. No es cero exacto porque τ arrastra la aritmética de
    los pasos anteriores, pero queda órdenes de magnitud por debajo del umbral."""
    config = Config()

    assert arco_de(quieta(config), config) == pytest.approx(0.0, abs=1e-9)


def test_el_arco_no_depende_de_la_distancia_a_la_camara() -> None:
    """τ ya está dividida por la escala de la mano, así que el mismo trazo mide lo
    mismo ejecutado cerca y lejos. Sin esa invariancia el umbral sería inútil:
    habría que ponerle uno distinto a cada persona y a cada silla."""
    config = Config()
    offsets = arc_offsets(24, width_px=120.0, depth_px=160.0)

    cerca = bufferizar(moving_sequence(canonical_hand(), offsets))
    lejos = bufferizar(
        moving_sequence(
            tuple((x / 2, y / 2, z / 2) for x, y, z in canonical_hand()),
            tuple((dx / 2, dy / 2) for dx, dy in offsets),
        )
    )

    assert arco_de(cerca, config) == pytest.approx(arco_de(lejos, config), rel=1e-9)


def test_una_dinamica_sin_recorrido_se_rechaza() -> None:
    """El agujero que hasta ahora no tapaba nada.

    Una "J" en la que la mano apenas se movió es, en el dataset, indistinguible de
    una "I": misma configuración de dedos, trayectoria plana. Envenena al
    clasificador dinámico por el lado contrario al que tapa la clase negativa, y
    mirando el archivo después no hay forma de saber cuál era cuál.
    """
    config = Config()
    frames = cambiando_de_forma(config.capture.static_frames)

    quality = evaluate_window(frames, SampleKind.DYNAMIC, config)

    assert quality.rejection is Rejection.TRAJECTORY_TOO_SHORT
    assert quality.arc_length is not None
    assert quality.arc_length < config.capture.min_trajectory_arc


def test_el_arco_se_informa_aunque_la_ventana_se_rechace() -> None:
    """Es lo que hace útil la barra del preview en modo dinámico: quien graba
    tiene que ver cuánto le falta, no solo que le falta."""
    config = Config()

    quality = evaluate_window(
        cambiando_de_forma(config.capture.static_frames), SampleKind.DYNAMIC, config
    )

    assert quality.arc_length is not None


def test_el_arco_no_decide_en_las_estaticas() -> None:
    """Una estática legítima tiene arco casi nulo por definición. Aplicarle el
    umbral de las dinámicas rechazaría el alfabeto entero menos ocho letras."""
    config = Config()

    quality = evaluate_window(quieta(config), SampleKind.STATIC, config)

    assert quality.accepted
    assert quality.arc_length is not None
    assert quality.arc_length < config.capture.min_trajectory_arc


def test_la_sigma_no_decide_en_las_dinamicas() -> None:
    """El espejo del test anterior."""
    config = Config()

    quality = evaluate_window(
        dinamica_realista(config.capture.static_frames), SampleKind.DYNAMIC, config
    )

    assert quality.accepted
    assert quality.dispersion is not None
    assert quality.dispersion > config.quality.max_dispersion


def test_una_grabacion_dinamica_pasada_de_frames_se_rechaza() -> None:
    """El tope de una dinámica es `dynamic_max_ms` a la tasa nominal (Bloque 4).
    Una estática no lo tiene: sale del buffer circular y mide lo que mide."""
    from lsm.segmentation import frames_from_ms

    config = Config()
    tope = frames_from_ms(config.capture.dynamic_max_ms, config.capture.camera_fps)
    frames = dinamica_realista(tope + 1)

    quality = evaluate_window(frames, SampleKind.DYNAMIC, config)

    assert quality.rejection is Rejection.TOO_MANY_FRAMES
    assert maximum_frames(SampleKind.DYNAMIC, config) == tope
    assert maximum_frames(SampleKind.STATIC, config) is None


def test_la_dispersion_no_mide_el_viaje_de_la_mano_sino_su_forma() -> None:
    """Propiedad poco intuitiva y conviene tenerla fijada por escrito.

    σ se calcula sobre el vector del §2, que ya pasó por la traslación del paso 3
    y la escala del paso 4. Una mano que cruza el encuadre entero sin cambiar de
    configuración tiene σ ≈ 0 y la ventana se acepta — y está bien, porque la
    seña es exactamente la misma esté donde esté la mano.

    Quien busque rechazar el *viaje* no debe tocar este umbral: lo que mide el
    desplazamiento es `v_t` del §6, que se calcula sobre puntos sin trasladar y es
    insumo de la segmentación, no de la captura.
    """
    config = Config()
    viajando = bufferizar(
        moving_sequence(
            canonical_hand(),
            arc_offsets(config.capture.static_frames, width_px=300.0, depth_px=300.0),
        )
    )

    quality = evaluate_window(viajando, SampleKind.STATIC, config)

    assert quality.accepted
    assert quality.dispersion == pytest.approx(0.0, abs=1e-12)


def test_una_dinamica_admite_menos_frames_que_una_estatica() -> None:
    config = Config()

    assert minimum_frames(SampleKind.DYNAMIC, config) == (
        config.capture.dynamic_min_frames
    )
    assert minimum_frames(SampleKind.STATIC, config) == config.capture.static_frames


def test_una_etiqueta_de_mediapipe_que_cambia_ya_no_rechaza_la_muestra() -> None:
    """Hasta el ADR 0017 esto era `MIXED_HANDEDNESS`: si MediaPipe cambiaba de
    etiqueta a mitad de muestra, se rechazaba. Con la mano **declarada** ese caso
    deja de existir: todos los frames llevan la mano de la sesión, y lo que diga
    MediaPipe viaja aparte en `detected_handedness` sin decidir nada. Rechazarlo
    sesgaba el dataset hacia los intentos en que MediaPipe no dudó —justo los de
    la J y la Q con la palma de frente—."""
    from dataclasses import replace

    config = Config()
    derecha = to_frame(canonical_hand(), width=1280, height=720)
    mitad = config.capture.static_frames // 2
    frames = tuple(
        BufferedFrame(
            slot=replace(
                derecha,
                detected_handedness=(
                    Handedness.RIGHT if indice < mitad else Handedness.LEFT
                ),
            ),
            luminance=LUZ,
        )
        for indice in range(config.capture.static_frames)
    )

    quality = evaluate_window(frames, SampleKind.STATIC, config)

    assert quality.rejection is None
    assert quality.handedness is Handedness.RIGHT


def test_una_mano_degenerada_se_rechaza_sin_dividir_por_cero() -> None:
    config = Config()
    frames = bufferizar(still_sequence(collapsed_scale(canonical_hand()), length=24))

    quality = evaluate_window(frames, SampleKind.STATIC, config)

    assert quality.rejection is Rejection.DEGENERATE_SCALE
    assert quality.mean_scale_px is None


def test_la_luminancia_de_la_ventana_es_la_media_de_sus_cuadros() -> None:
    config = Config()
    frames = tuple(
        BufferedFrame(slot=buffered.slot, luminance=0.2 if indice % 2 else 0.6)
        for indice, buffered in enumerate(quieta(config))
    )

    quality = evaluate_window(frames, SampleKind.STATIC, config)

    assert quality.mean_luminance == pytest.approx(0.4)


def test_toda_causa_de_rechazo_sabe_decir_que_corregir() -> None:
    """Un rechazo sin instrucción no sirve de nada a los cuarenta minutos de
    sesión. Si alguien añade un motivo nuevo, este test se lo recuerda."""
    for rejection in Rejection:
        mensaje = explain(rejection)
        assert mensaje
        assert mensaje.isascii(), (
            f"{rejection}: la fuente de OpenCV es ASCII y un acento sale como '?'"
        )


# --------------------------------------------------------------------------- #
# El espejado del preview
# --------------------------------------------------------------------------- #


def test_el_preview_espejado_refleja_x_y_deja_y_en_su_sitio() -> None:
    """`feature-spec.md` §0.3 en su forma práctica.

    El detector ve el mundo sin espejar; la pantalla lo ve reflejado. Si `y` se
    invirtiera también, la mano aparecería boca abajo: los landmarks ya vienen en
    convención de imagen, con el eje apuntando hacia abajo, igual que los píxeles.
    """
    landmark = Landmark(x=0.25, y=0.75, z=0.0)

    assert preview_position(landmark, 1280, 720, mirrored=True) == (960, 540)
    assert preview_position(landmark, 1280, 720, mirrored=False) == (320, 540)


def test_el_espejado_es_su_propio_inverso() -> None:
    for x in (0.0, 0.1, 0.5, 0.9, 1.0):
        landmark = Landmark(x=x, y=0.5, z=0.0)
        espejado = preview_position(landmark, 1000, 1000, mirrored=True)
        vuelta = preview_position(
            Landmark(x=espejado[0] / 1000, y=0.5, z=0.0), 1000, 1000, mirrored=True
        )
        assert vuelta[0] == round(x * 1000)


def test_un_landmark_fuera_del_encuadre_no_se_recorta() -> None:
    """MediaPipe extrapola fuera del cuadro a propósito. Un dedo que se sale por
    el borde debe dibujarse saliéndose, no pegado al borde fingiendo que sigue
    dentro: si se recortara, el preview mentiría justo cuando hay que corregir el
    encuadre."""
    fuera = Landmark(x=1.4, y=-0.2, z=0.0)

    x, y = preview_position(fuera, 1000, 500, mirrored=False)

    assert (x, y) == (1400, -100)


# --------------------------------------------------------------------------- #
# Bloque 4: la dinámica la delimita la máquina de estados
# --------------------------------------------------------------------------- #


def _trazo_y_reposo(movimiento: int, reposo: int) -> list[RawFrame]:
    """Una mano quieta, que se mueve `movimiento` cuadros y se detiene."""
    from lsm.synthetic import canonical_hand, to_frame, translated

    base = canonical_hand()
    quieta_antes = [
        to_frame(translated(base, 400.0, 400.0), width=1280, height=720)
    ] * 10
    trazo = [
        to_frame(translated(base, 400.0 + 12.0 * i, 400.0), width=1280, height=720)
        for i in range(1, movimiento + 1)
    ]
    quieta = [trazo[-1]] * reposo
    return [*quieta_antes, *trazo, *quieta]


def test_la_muestra_dinamica_es_el_trazo_de_la_maquina_mas_su_reposo() -> None:
    from lsm.capture import DelimitedStroke, delimit_dynamic_stroke
    from lsm.segmentation import FrameThresholds

    # Sin One Euro: su retraso corre el frame de partida uno (§4).
    config = Config.model_validate({"smoothing": {"enabled": False}})
    fps = float(config.capture.camera_fps)
    umbrales = FrameThresholds.from_config(config, fps)
    flujo = _trazo_y_reposo(
        movimiento=20, reposo=umbrales.motion_confirm_low_frames + 10
    )

    trazo = delimit_dynamic_stroke(iter(flujo), config, fps)

    assert isinstance(trazo, DelimitedStroke)
    # Empieza en el frame de partida del movimiento (el último quieto, índice 9)
    # y se entrega tras el reposo que lo cierra.
    assert trazo.start == 9
    assert trazo.end > trazo.start + trazo.stroke_frames
    assert trazo.stroke_frames >= 20


def test_si_la_mano_no_se_detiene_la_dinamica_no_se_guarda() -> None:
    """Se acaba el flujo —el tope de tiempo— con el trazo en marcha: nunca más
    un trazo truncado (Bloque 4)."""
    from lsm.capture import delimit_dynamic_stroke

    config = Config()
    flujo = _trazo_y_reposo(movimiento=30, reposo=0)

    assert (
        delimit_dynamic_stroke(iter(flujo), config, 30.0) is Rejection.STROKE_NOT_CLOSED
    )


def test_un_trazo_que_no_termina_nunca_se_rechaza_por_largo() -> None:
    from lsm.capture import delimit_dynamic_stroke
    from lsm.segmentation import FrameThresholds

    config = Config()
    umbrales = FrameThresholds.from_config(config, 30.0)
    flujo = _trazo_y_reposo(movimiento=umbrales.motion_max_frames + 20, reposo=0)

    assert (
        delimit_dynamic_stroke(iter(flujo), config, 30.0) is Rejection.STROKE_TOO_LONG
    )


def test_un_trazo_cortado_por_un_hueco_largo_se_rechaza_por_interrumpido() -> None:
    """Paso 0: el motivo dice que se perdió la mano, no que no se detuvo."""
    from lsm.capture import delimit_dynamic_stroke
    from lsm.segmentation import FrameThresholds

    config = Config()
    umbrales = FrameThresholds.from_config(config, 30.0)
    hueco_largo = [InvalidFrame(reason=InvalidReason.NO_HAND)] * (
        umbrales.missing_frames_to_idle - 1
    )
    flujo: list[FrameSlot] = [*_trazo_y_reposo(movimiento=20, reposo=0), *hueco_largo]

    assert (
        delimit_dynamic_stroke(iter(flujo), config, 30.0)
        is Rejection.STROKE_INTERRUPTED
    )


def test_un_trazo_demasiado_reconstruido_se_rechaza_por_exceso_de_relleno() -> None:
    """Paso 0: la máquina cierra el trazo pero lo descarta por relleno."""
    from lsm.capture import delimit_dynamic_stroke
    from lsm.segmentation import FrameThresholds
    from lsm.synthetic import canonical_hand, to_frame, translated

    config = Config()
    umbrales = FrameThresholds.from_config(config, 30.0)
    base = canonical_hand()
    flujo: list[RawFrame | InvalidFrame] = [
        to_frame(translated(base, 400.0, 400.0), width=1280, height=720)
    ] * 10
    for i in range(1, 60):
        mano = to_frame(
            translated(base, 400.0 + 12.0 * i, 400.0), width=1280, height=720
        )
        # Tras entrar al candidato, uno de cada dos cuadros se pierde: la mitad
        # del trazo sale interpolada.
        flujo.append(
            InvalidFrame(reason=InvalidReason.NO_HAND) if i > 20 and i % 2 else mano
        )
    ultimo = flujo[-1] if isinstance(flujo[-1], RawFrame) else flujo[-2]
    flujo += [ultimo] * (umbrales.motion_confirm_low_frames + 10)

    assert (
        delimit_dynamic_stroke(iter(flujo), config, 30.0)
        is Rejection.TOO_MUCH_INTERPOLATED
    )


def test_un_intento_rechazado_se_vuelve_a_segmentar_igual_que_en_vivo() -> None:
    """`resegment_attempt` es la captura sin cámara: el mismo trazo, la misma
    evaluación. Un intento que se perdió por un hueco sigue perdido con la
    configuración con que se perdió."""
    from lsm.capture import ResegmentedStroke, resegment_attempt
    from lsm.segmentation import FrameThresholds

    config = Config.model_validate({"smoothing": {"enabled": False}})
    umbrales = FrameThresholds.from_config(config, 30.0)
    bueno = tuple(
        _trazo_y_reposo(movimiento=20, reposo=umbrales.motion_confirm_low_frames + 5)
    )
    resultado = resegment_attempt(bueno, config, 30.0)
    assert isinstance(resultado, ResegmentedStroke)
    assert resultado.stroke.start == 9
    assert resultado.quality.accepted

    cortado: tuple[FrameSlot, ...] = (
        *_trazo_y_reposo(movimiento=20, reposo=0),
        *[InvalidFrame(reason=InvalidReason.NO_HAND)] * 8,
    )
    assert resegment_attempt(cortado, config, 30.0) is Rejection.STROKE_INTERRUPTED


def test_una_dinamica_que_acaba_en_movimiento_es_truncada() -> None:
    from lsm.capture import is_truncated

    config = Config()

    assert is_truncated(tuple(_trazo_y_reposo(movimiento=20, reposo=0)), config)
    assert not is_truncated(tuple(_trazo_y_reposo(movimiento=20, reposo=20)), config)
