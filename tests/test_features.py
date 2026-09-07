"""Tests del contrato de `docs/feature-spec.md`.

Los tests de invariancia son el corazón del proyecto. No verifican que el código
corra sin excepciones: verifican que la misma seña, hecha con la otra mano, más
cerca, en otra esquina del encuadre, con la muñeca torcida o en un frame de otra
relación de aspecto, produzca **el mismo vector**. Si alguna de esas invariancias
no se cumple, el modelo funciona para quien lo entrenó y para nadie más.

Los landmarks son sintéticos y se construyen en píxeles con transformaciones
conocidas (`lsm.synthetic`), porque mover la mano en el mundo es lo que hay que
simular; mover el vector normalizado no probaría nada.
"""

from __future__ import annotations

import math

import pytest

from lsm.config import Config
from lsm.features import (
    FEATURE_SPEC_VERSION,
    MIN_SCALE,
    RESAMPLE_LENGTH,
    DynamicRejection,
    DynamicUnavailable,
    ExtractionRejected,
    SequenceFeatures,
    canonicalize_handedness,
    correct_aspect_and_orientation,
    extract_sequence_features,
    flatten,
    mean_displacement,
    reference_scale,
    resample,
    rotate_to_axis,
    smooth_sequence,
    split_valid_runs,
    translate_to_origin,
)
from lsm.synthetic import (
    arc_offsets,
    canonical_hand,
    collapsed_scale,
    fist_hand,
    mirrored_x,
    moving_sequence,
    rotated,
    scaled,
    still_sequence,
    to_frame,
    translated,
)
from lsm.types import (
    NUM_FEATURES,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    LandmarkIndex,
    RawFrame,
    Sequence,
)

#: Las invariancias son exactas salvo por el redondeo de punto flotante. El
#: contrato con TypeScript tolera 1e-6; aquí se aprieta tres órdenes de magnitud
#: porque comparamos Python contra Python y un error real no es de redondeo.
TOL = 1e-9

CONFIG = Config()


def features_of(sequence: Sequence) -> tuple[float, ...]:
    """Vector de forma del §2 de una secuencia, o falla el test si se rechazó."""
    outcome = extract_sequence_features(sequence, CONFIG)
    assert isinstance(outcome, SequenceFeatures), f"extracción rechazada: {outcome}"
    return outcome.static.shape.values


def assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert a == pytest.approx(b, abs=TOL), f"componente {index}: {a} != {b}"


def base_sequence(length: int = 6) -> Sequence:
    """La seña de referencia: mano derecha, vertical, centrada, frame 16:9."""
    return still_sequence(translated(canonical_hand(), 640.0, 400.0), length=length)


# --------------------------------------------------------------------------- #
# Camino base y postcondiciones de cada paso
# --------------------------------------------------------------------------- #


def test_el_vector_tiene_42_componentes_y_lleva_la_version_del_spec() -> None:
    outcome = extract_sequence_features(base_sequence(), CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert len(outcome.static.shape) == NUM_FEATURES
    assert outcome.static.shape.spec_version == FEATURE_SPEC_VERSION
    assert FEATURE_SPEC_VERSION == 1


def test_tras_el_pipeline_la_muneca_queda_en_el_origen_y_p9_en_0_1() -> None:
    """Postcondición de los pasos 3 y 5 del §1, y razón de que se conserven
    componentes constantes en el vector: mantienen alineada la numeración."""
    values = features_of(base_sequence())

    wrist = 2 * int(LandmarkIndex.WRIST)
    middle_mcp = 2 * int(LandmarkIndex.MIDDLE_MCP)

    assert values[wrist] == pytest.approx(0.0, abs=TOL)
    assert values[wrist + 1] == pytest.approx(0.0, abs=TOL)
    assert values[middle_mcp] == pytest.approx(0.0, abs=TOL)
    assert values[middle_mcp + 1] == pytest.approx(1.0, abs=TOL)


def test_el_paso_1_corrige_el_aspecto_y_voltea_el_eje_y() -> None:
    points = ((0.5, 0.25, 0.1),) * 21
    corrected = correct_aspect_and_orientation(points, 16 / 9)

    assert corrected[0][0] == pytest.approx(0.5 * 16 / 9)
    assert corrected[0][1] == pytest.approx(-0.25)
    assert corrected[0][2] == pytest.approx(0.1 * 16 / 9)


def test_el_paso_2_solo_toca_la_mano_izquierda() -> None:
    points = ((0.3, -0.2, 0.05),) * 21

    assert canonicalize_handedness(points, Handedness.RIGHT) == points
    assert canonicalize_handedness(points, Handedness.LEFT)[0] == (-0.3, -0.2, 0.05)


def test_el_paso_3_deja_la_muneca_exactamente_en_cero() -> None:
    points = ((1.5, 2.5, 0.5), (2.0, 3.0, 0.75)) + ((0.0, 0.0, 0.0),) * 19

    translated_points = translate_to_origin(points)

    assert translated_points[0] == (0.0, 0.0, 0.0)
    assert translated_points[1] == (0.5, 0.5, 0.25)


def test_el_paso_4_ignora_z_al_medir_la_escala() -> None:
    points = tuple((0.0, 0.0, 0.0) for _ in range(21))
    points = (*points[:9], (3.0, 4.0, 99.0), *points[10:])

    assert reference_scale(points) == pytest.approx(5.0)


def test_el_paso_5_alinea_p9_con_el_eje_y() -> None:
    points = tuple((0.0, 0.0, 0.0) for _ in range(21))
    points = (*points[:9], (1.0, 0.0, 0.3), *points[10:])

    rotated_points = rotate_to_axis(points)

    assert rotated_points[9][0] == pytest.approx(0.0, abs=TOL)
    assert rotated_points[9][1] == pytest.approx(1.0, abs=TOL)
    assert rotated_points[9][2] == 0.3, "el paso 5 no debe tocar z"


def test_el_paso_7_aplana_por_indice_ascendente_x_antes_que_y() -> None:
    points_2d = tuple((float(i), float(i) + 0.5) for i in range(21))

    vector = flatten(points_2d)

    assert vector.values[0] == 0.0
    assert vector.values[1] == 0.5
    assert vector.values[2] == 1.0
    assert vector.values[41] == 20.5


# --------------------------------------------------------------------------- #
# Invariancias — la tabla de feature-spec.md §5.1
# --------------------------------------------------------------------------- #


def test_invariancia_de_lateralidad_izquierda_da_lo_mismo_que_derecha() -> None:
    """Paso 2. Una persona zurda debe poder usar un modelo entrenado por diestros."""
    right = base_sequence()
    left = still_sequence(
        mirrored_x(translated(canonical_hand(), 640.0, 400.0)),
        length=6,
        handedness=Handedness.LEFT,
    )

    assert_close(features_of(left), features_of(right))


def test_invariancia_de_escala_la_misma_sena_al_doble_de_tamano() -> None:
    """Paso 4. Entrenar a 50 cm y firmar a 1.5 m debe dar el mismo vector."""
    near = still_sequence(
        scaled(translated(canonical_hand(), 640.0, 400.0), 2.0), length=6
    )

    assert_close(features_of(near), features_of(base_sequence()))


def test_invariancia_de_escala_a_la_mitad_de_tamano() -> None:
    far = still_sequence(
        scaled(translated(canonical_hand(), 640.0, 400.0), 0.5), length=6
    )

    assert_close(features_of(far), features_of(base_sequence()))


@pytest.mark.parametrize(
    "position",
    [(120.0, 260.0), (1160.0, 260.0), (120.0, 660.0), (1160.0, 660.0)],
    ids=["arriba_izq", "arriba_der", "abajo_izq", "abajo_der"],
)
def test_invariancia_de_posicion_en_las_cuatro_esquinas(
    position: tuple[float, float],
) -> None:
    """Paso 3. Dónde esté la mano en el encuadre no cambia qué letra es."""
    moved = still_sequence(translated(canonical_hand(), *position), length=6)

    assert_close(features_of(moved), features_of(base_sequence()))


@pytest.mark.parametrize("degrees", [45.0, 90.0, -45.0, 135.0])
def test_invariancia_de_rotacion(degrees: float) -> None:
    """Paso 5. La inclinación de la muñeca no cambia la letra."""
    tilted = still_sequence(
        rotated(translated(canonical_hand(), 640.0, 400.0), degrees), length=6
    )

    assert_close(features_of(tilted), features_of(base_sequence()))


def test_invariancia_de_relacion_de_aspecto_16_9_contra_1_1() -> None:
    """Paso 1. La misma mano, en píxeles, filmada con dos encuadres distintos."""
    hand = translated(canonical_hand(), 360.0, 400.0)
    wide = still_sequence(hand, length=6, width=1280, height=720)
    square = still_sequence(hand, length=6, width=720, height=720)

    assert_close(features_of(square), features_of(wide))


def test_invariancia_de_relacion_de_aspecto_frame_vertical() -> None:
    hand = translated(canonical_hand(), 360.0, 400.0)
    portrait = still_sequence(hand, length=6, width=720, height=1280)
    wide = still_sequence(hand, length=6, width=1280, height=720)

    assert_close(features_of(portrait), features_of(wide))


def test_las_invariancias_se_componen() -> None:
    """Mano izquierda, girada, al doble de tamaño, en otra esquina y en 1:1."""
    hand = mirrored_x(rotated(scaled(canonical_hand(), 2.0), 30.0))
    combined = still_sequence(
        translated(hand, 300.0, 500.0),
        length=6,
        width=1000,
        height=1000,
        handedness=Handedness.LEFT,
    )

    assert_close(features_of(combined), features_of(base_sequence()))


def test_dos_configuraciones_distintas_no_dan_el_mismo_vector() -> None:
    """Control negativo: sin esto, un `return zeros(42)` pasaría todo lo anterior."""
    open_hand = features_of(base_sequence())
    fist = features_of(still_sequence(translated(fist_hand(), 640.0, 400.0), length=6))

    assert max(abs(a - b) for a, b in zip(open_hand, fist, strict=True)) > 0.1


# --------------------------------------------------------------------------- #
# Casos degenerados
# --------------------------------------------------------------------------- #


def test_escala_nula_marca_la_secuencia_invalida_en_vez_de_dividir_por_cero() -> None:
    degenerate = still_sequence(
        collapsed_scale(translated(canonical_hand(), 640.0, 400.0)), length=4
    )

    outcome = extract_sequence_features(degenerate, CONFIG)

    assert isinstance(outcome, ExtractionRejected)
    assert outcome.reason is InvalidReason.SCALE_TOO_SMALL
    assert outcome.frame_index == 0


def test_la_escala_degenerada_se_reporta_en_el_frame_donde_ocurre() -> None:
    good = to_frame(translated(canonical_hand(), 640.0, 400.0), width=1280, height=720)
    bad = to_frame(
        collapsed_scale(translated(canonical_hand(), 640.0, 400.0)),
        width=1280,
        height=720,
    )
    sequence = Sequence(frames=(good, good, bad, good))

    outcome = extract_sequence_features(sequence, CONFIG)

    assert isinstance(outcome, ExtractionRejected)
    assert outcome.frame_index == 2


def test_una_escala_apenas_bajo_el_minimo_tambien_se_rechaza() -> None:
    points = list(canonical_hand())
    points[LandmarkIndex.MIDDLE_MCP] = (MIN_SCALE * 1280 * 0.5, 0.0, 0.0)
    sequence = still_sequence(tuple(points), length=3)

    assert isinstance(extract_sequence_features(sequence, CONFIG), ExtractionRejected)


def test_landmarks_fuera_de_cero_uno_no_rompen_nada() -> None:
    """MediaPipe extrapola fuera del encuadre; eso no es un error."""
    outside = still_sequence(translated(canonical_hand(), -80.0, 40.0), length=6)

    frame = outside.frames[0]
    assert min(landmark.x for landmark in frame.landmarks) < 0.0
    assert min(landmark.y for landmark in frame.landmarks) < 0.0

    assert_close(features_of(outside), features_of(base_sequence()))


def test_cero_manos_detectadas_produce_un_marcador_no_un_none() -> None:
    stream: FrameStream = (InvalidFrame(reason=InvalidReason.NO_HAND),)

    assert split_valid_runs(stream) == ()


# --------------------------------------------------------------------------- #
# Secuencias interrumpidas: el inicio, el medio y el final se comportan distinto
# --------------------------------------------------------------------------- #


def _stream_frames(count: int) -> tuple[RawFrame, ...]:
    return tuple(
        to_frame(
            translated(canonical_hand(), 600.0 + 12.0 * i, 400.0),
            width=1280,
            height=720,
        )
        for i in range(count)
    )


def test_un_frame_invalido_al_inicio_recorta_la_secuencia() -> None:
    frames = _stream_frames(5)
    stream: FrameStream = (InvalidFrame(reason=InvalidReason.NO_HAND), *frames)

    runs = split_valid_runs(stream)

    assert len(runs) == 1
    assert len(runs[0]) == 5


def test_un_frame_invalido_al_final_recorta_la_secuencia() -> None:
    frames = _stream_frames(5)
    stream: FrameStream = (*frames, InvalidFrame(reason=InvalidReason.NO_HAND))

    runs = split_valid_runs(stream)

    assert len(runs) == 1
    assert len(runs[0]) == 5


def test_un_frame_invalido_en_medio_parte_la_secuencia_en_dos() -> None:
    """§0.3: los frames inválidos no se interpolan, interrumpen la secuencia."""
    frames = _stream_frames(6)
    stream: FrameStream = (
        *frames[:3],
        InvalidFrame(reason=InvalidReason.NO_HAND),
        *frames[3:],
    )

    runs = split_valid_runs(stream)

    assert len(runs) == 2
    assert [len(run) for run in runs] == [3, 3]


def test_perder_el_primer_frame_mueve_el_origen_de_la_trayectoria() -> None:
    """Por eso importa dónde cae el frame inválido y no solo cuántos hay.

    El origen del canal de trayectoria es la muñeca del **primer frame válido**:
    si se pierde el primero, el trazo se mide desde otro punto.
    """
    frames = _stream_frames(5)
    full = extract_sequence_features(Sequence(frames=frames), CONFIG)
    truncated = extract_sequence_features(Sequence(frames=frames[1:]), CONFIG)

    assert isinstance(full, SequenceFeatures)
    assert isinstance(truncated, SequenceFeatures)
    assert full.trajectory.points[-1][0] != pytest.approx(
        truncated.trajectory.points[-1][0], abs=TOL
    )


# --------------------------------------------------------------------------- #
# §2 — agregación estática
# --------------------------------------------------------------------------- #


def test_una_sena_estable_tiene_dispersion_practicamente_cero() -> None:
    """No se exige cero exacto: con T frames idénticos el promedio ya arrastra
    redondeo, así que σ queda en el orden de 1e-17. Lo que importa es que esté
    muchísimo por debajo de `quality.max_dispersion`."""
    outcome = extract_sequence_features(base_sequence(length=10), CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert outcome.static.dispersion < 1e-12
    assert outcome.static.dispersion < CONFIG.quality.max_dispersion


def test_la_dispersion_usa_desviacion_poblacional_ddof_cero() -> None:
    """Fijado explícitamente: σ no viaja en los golden vectors de un frame, así
    que una discrepancia ddof=0 / ddof=1 con TypeScript sería invisible."""
    open_frame = to_frame(
        translated(canonical_hand(), 640.0, 400.0), width=1280, height=720
    )
    fist_frame = to_frame(translated(fist_hand(), 640.0, 400.0), width=1280, height=720)
    sequence = Sequence(frames=(open_frame, fist_frame))

    outcome = extract_sequence_features(sequence, CONFIG)
    assert isinstance(outcome, SequenceFeatures)

    a = outcome.frames[0].values
    b = outcome.frames[1].values
    expected = sum(abs(x - y) / 2.0 for x, y in zip(a, b, strict=True)) / NUM_FEATURES

    assert outcome.static.dispersion == pytest.approx(expected, abs=1e-15)


def test_el_vector_de_forma_es_el_promedio_temporal() -> None:
    open_frame = to_frame(
        translated(canonical_hand(), 640.0, 400.0), width=1280, height=720
    )
    fist_frame = to_frame(translated(fist_hand(), 640.0, 400.0), width=1280, height=720)
    sequence = Sequence(frames=(open_frame, fist_frame))

    outcome = extract_sequence_features(sequence, CONFIG)
    assert isinstance(outcome, SequenceFeatures)

    expected = tuple(
        (x + y) / 2.0
        for x, y in zip(outcome.frames[0].values, outcome.frames[1].values, strict=True)
    )
    assert_close(outcome.static.shape.values, expected)


def test_la_dispersion_no_es_una_feature() -> None:
    outcome = extract_sequence_features(base_sequence(), CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert len(outcome.static.shape) == NUM_FEATURES


# --------------------------------------------------------------------------- #
# §3 — canal de trayectoria: la razón de ser de todo el §3
# --------------------------------------------------------------------------- #


def test_la_trayectoria_arranca_en_el_origen() -> None:
    outcome = extract_sequence_features(
        moving_sequence(canonical_hand(), arc_offsets(8)), CONFIG
    )

    assert isinstance(outcome, SequenceFeatures)
    assert outcome.trajectory.points[0] == (0.0, 0.0)


def test_una_mano_quieta_no_traza_nada() -> None:
    outcome = extract_sequence_features(base_sequence(length=8), CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert all(point == (0.0, 0.0) for point in outcome.trajectory.points)


def test_la_trayectoria_captura_el_movimiento_que_el_paso_3_destruye() -> None:
    """El test que separa una J de una I.

    Las dos secuencias tienen la **misma configuración de mano en cada frame**.
    Solo cambia que una se mueve. Si el canal de trayectoria se computara después
    de la traslación del paso 3, saldría idénticamente cero y las dos secuencias
    serían indistinguibles.
    """
    still = extract_sequence_features(base_sequence(length=8), CONFIG)
    moving = extract_sequence_features(
        moving_sequence(canonical_hand(), arc_offsets(8)), CONFIG
    )
    assert isinstance(still, SequenceFeatures)
    assert isinstance(moving, SequenceFeatures)

    # Misma forma, frame a frame.
    for still_frame, moving_frame in zip(still.frames, moving.frames, strict=True):
        assert_close(still_frame.values, moving_frame.values)
    assert_close(still.static.shape.values, moving.static.shape.values)

    # Trazo distinto.
    displacement = max(math.hypot(x, y) for x, y in moving.trajectory.points)
    assert displacement > 0.5


def test_la_trayectoria_es_invariante_a_la_posicion_en_el_encuadre() -> None:
    offsets = arc_offsets(8)
    centered = extract_sequence_features(
        moving_sequence(canonical_hand(), offsets), CONFIG
    )
    cornered = extract_sequence_features(
        moving_sequence(translated(canonical_hand(), 420.0, 180.0), offsets), CONFIG
    )
    assert isinstance(centered, SequenceFeatures)
    assert isinstance(cornered, SequenceFeatures)

    for a, b in zip(
        centered.trajectory.points, cornered.trajectory.points, strict=True
    ):
        assert a[0] == pytest.approx(b[0], abs=TOL)
        assert a[1] == pytest.approx(b[1], abs=TOL)


def test_la_trayectoria_es_invariante_a_la_distancia_a_la_camara() -> None:
    """La misma seña ejecutada más cerca: mano al doble y trazo al doble."""
    offsets = arc_offsets(8)
    far = extract_sequence_features(moving_sequence(canonical_hand(), offsets), CONFIG)
    near = extract_sequence_features(
        moving_sequence(
            scaled(canonical_hand(), 2.0),
            tuple((dx * 2.0, dy * 2.0) for dx, dy in offsets),
        ),
        CONFIG,
    )
    assert isinstance(far, SequenceFeatures)
    assert isinstance(near, SequenceFeatures)

    for a, b in zip(far.trajectory.points, near.trajectory.points, strict=True):
        assert a[0] == pytest.approx(b[0], abs=TOL)
        assert a[1] == pytest.approx(b[1], abs=TOL)


def test_la_trayectoria_se_mide_en_unidades_de_mano() -> None:
    """Un desplazamiento de un largo de mano debe dar τ ≈ 1."""
    hand = canonical_hand()
    scale_px = math.hypot(hand[9][0] - hand[0][0], hand[9][1] - hand[0][1])
    outcome = extract_sequence_features(
        moving_sequence(hand, ((0.0, 0.0), (scale_px, 0.0))), CONFIG
    )

    assert isinstance(outcome, SequenceFeatures)
    assert outcome.trajectory.points[1][0] == pytest.approx(1.0, abs=1e-12)


def test_el_eje_y_de_la_trayectoria_apunta_hacia_arriba() -> None:
    """Coherente con el paso 1: bajar la mano en pantalla es τ_y negativo."""
    outcome = extract_sequence_features(
        moving_sequence(canonical_hand(), ((0.0, 0.0), (0.0, 50.0))), CONFIG
    )

    assert isinstance(outcome, SequenceFeatures)
    assert outcome.trajectory.points[1][1] < 0.0


def test_la_lateralidad_se_canoniza_tambien_en_la_trayectoria() -> None:
    """Un trazo hacia la derecha con la mano izquierda es, tras canonizar, un
    trazo hacia la izquierda: la J zurda es el espejo de la J diestra."""
    right = extract_sequence_features(
        moving_sequence(canonical_hand(), ((0.0, 0.0), (60.0, 0.0))), CONFIG
    )
    left = extract_sequence_features(
        moving_sequence(
            mirrored_x(canonical_hand()),
            ((0.0, 0.0), (-60.0, 0.0)),
            handedness=Handedness.LEFT,
        ),
        CONFIG,
    )
    assert isinstance(right, SequenceFeatures)
    assert isinstance(left, SequenceFeatures)

    assert left.trajectory.points[1][0] == pytest.approx(
        right.trajectory.points[1][0], abs=TOL
    )


# --------------------------------------------------------------------------- #
# §3.2 — remuestreo
# --------------------------------------------------------------------------- #


def test_el_remuestreo_conserva_los_extremos_exactamente() -> None:
    rows = tuple((float(i), float(i) * 2.0) for i in range(7))

    out = resample(rows, RESAMPLE_LENGTH)

    assert len(out) == RESAMPLE_LENGTH
    assert out[0] == rows[0]
    assert out[-1] == rows[-1]


def test_el_remuestreo_interpola_linealmente() -> None:
    rows = ((0.0,), (1.0,))

    out = resample(rows, 3)

    assert out[1][0] == pytest.approx(0.5)


def test_una_secuencia_ya_de_24_frames_se_remuestrea_a_si_misma() -> None:
    rows = tuple((float(i),) for i in range(RESAMPLE_LENGTH))

    assert resample(rows, RESAMPLE_LENGTH) == rows


def test_un_solo_frame_se_replica() -> None:
    out = resample(((3.0, -1.0),), RESAMPLE_LENGTH)

    assert len(out) == RESAMPLE_LENGTH
    assert all(row == (3.0, -1.0) for row in out)


def test_el_canal_dinamico_tiene_forma_24_por_44() -> None:
    outcome = extract_sequence_features(
        moving_sequence(canonical_hand(), arc_offsets(9)), CONFIG
    )

    assert isinstance(outcome, SequenceFeatures)
    dynamic = outcome.dynamic
    assert not isinstance(dynamic, DynamicUnavailable)
    assert len(dynamic.rows) == RESAMPLE_LENGTH
    assert all(len(row) == NUM_FEATURES + 2 for row in dynamic.rows)


def test_el_canal_dinamico_pondera_la_trayectoria() -> None:
    """§3.3: g_t = concat(f_t, w_τ · τ_t). Sin el peso, 2 componentes se pierden
    frente a 42 y el trazo se vuelve invisible en la distancia euclidiana."""
    sequence = moving_sequence(canonical_hand(), arc_offsets(9))
    outcome = extract_sequence_features(sequence, CONFIG)
    assert isinstance(outcome, SequenceFeatures)
    dynamic = outcome.dynamic
    assert not isinstance(dynamic, DynamicUnavailable)

    weight = CONFIG.features.trajectory_weight
    resampled_trajectory = resample(outcome.trajectory.points, RESAMPLE_LENGTH)
    for row, point in zip(dynamic.rows, resampled_trajectory, strict=True):
        assert row[NUM_FEATURES] == pytest.approx(weight * point[0], abs=TOL)
        assert row[NUM_FEATURES + 1] == pytest.approx(weight * point[1], abs=TOL)


def test_una_secuencia_demasiado_corta_no_se_estira_a_24() -> None:
    """Interpolar 2 frames a 24 inventa una trayectoria que nadie ejecutó."""
    short = moving_sequence(canonical_hand(), ((0.0, 0.0), (10.0, 0.0)))

    outcome = extract_sequence_features(short, CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert isinstance(outcome.dynamic, DynamicUnavailable)
    assert outcome.dynamic.reason is DynamicRejection.TOO_FEW_SOURCE_FRAMES


def test_una_secuencia_corta_conserva_sus_features_estaticas() -> None:
    """Rechazar el canal dinámico no invalida la seña: una letra estática puede
    resolverse con dos frames."""
    short = moving_sequence(canonical_hand(), ((0.0, 0.0), (10.0, 0.0)))

    outcome = extract_sequence_features(short, CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert len(outcome.static.shape) == NUM_FEATURES


# --------------------------------------------------------------------------- #
# §4 — suavizado
# --------------------------------------------------------------------------- #


def test_alpha_uno_desactiva_el_suavizado() -> None:
    sequence = moving_sequence(canonical_hand(), arc_offsets(6))

    assert smooth_sequence(sequence, 1.0) == sequence


def test_el_suavizado_conserva_el_primer_frame_y_arrastra_los_siguientes() -> None:
    sequence = moving_sequence(canonical_hand(), ((0.0, 0.0), (100.0, 0.0)))

    smoothed = smooth_sequence(sequence, 0.5)

    assert smoothed.frames[0] == sequence.frames[0]
    original_x = sequence.frames[1].landmarks[0].x
    previous_x = sequence.frames[0].landmarks[0].x
    assert smoothed.frames[1].landmarks[0].x == pytest.approx(
        0.5 * original_x + 0.5 * previous_x
    )


def test_el_suavizado_no_altera_los_metadatos_del_frame() -> None:
    sequence = moving_sequence(canonical_hand(), arc_offsets(4))

    smoothed = smooth_sequence(sequence, 0.4)

    assert smoothed.frames[1].width == sequence.frames[1].width
    assert smoothed.frames[1].handedness is sequence.frames[1].handedness


# --------------------------------------------------------------------------- #
# Velocidad — insumo de la máquina de estados
# --------------------------------------------------------------------------- #


def test_el_desplazamiento_medio_es_cero_entre_frames_identicos() -> None:
    points = tuple((float(i), float(i), 0.0) for i in range(21))

    assert mean_displacement(points, points) == 0.0


def test_el_desplazamiento_medio_ignora_z() -> None:
    a = tuple((0.0, 0.0, 0.0) for _ in range(21))
    b = tuple((3.0, 4.0, 100.0) for _ in range(21))

    assert mean_displacement(a, b) == pytest.approx(5.0)


def test_una_mano_quieta_tiene_velocidad_cero() -> None:
    outcome = extract_sequence_features(base_sequence(length=5), CONFIG)

    assert isinstance(outcome, SequenceFeatures)
    assert outcome.velocities == (0.0, 0.0, 0.0, 0.0)


def test_la_velocidad_se_mide_en_unidades_de_mano_y_no_en_pixeles() -> None:
    """Invariante a la distancia: la misma seña más cerca no parece más rápida."""
    hand = canonical_hand()
    offsets = ((0.0, 0.0), (20.0, 0.0), (40.0, 0.0))
    far = extract_sequence_features(moving_sequence(hand, offsets), CONFIG)
    near = extract_sequence_features(
        moving_sequence(
            scaled(hand, 2.0), tuple((dx * 2.0, dy * 2.0) for dx, dy in offsets)
        ),
        CONFIG,
    )
    assert isinstance(far, SequenceFeatures)
    assert isinstance(near, SequenceFeatures)

    for a, b in zip(far.velocities, near.velocities, strict=True):
        assert a == pytest.approx(b, abs=TOL)
    assert far.velocities[0] > 0.0
