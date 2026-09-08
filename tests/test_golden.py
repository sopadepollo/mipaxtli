"""El archivo de golden vectors es un contrato, y este test lo hace exigible.

Cumple dos funciones distintas:

1. **Regresión.** Si alguien cambia `features.py` sin regenerar el archivo, aquí
   revienta. Es el mecanismo que obliga a seguir la regla de `CLAUDE.md` §4:
   tocar la extracción implica incrementar la versión del spec y regenerar los
   fixtures.
2. **Espejo del test de TypeScript.** `web/tests/features.test.ts` hará
   exactamente esto mismo sobre el mismo archivo. Si este test es débil, aquel
   también lo será.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import lsm.features
import lsm.segmentation
from lsm.cli.golden import BASE_ID, MINIMUM_CASES
from lsm.config import Config
from lsm.features import (
    FEATURE_SPEC_VERSION,
    DynamicUnavailable,
    ExtractionRejected,
    SequenceFeatures,
    extract_sequence_features,
    split_valid_runs,
)
from lsm.segmentation import SEGMENTATION_SPEC_VERSION
from lsm.types import (
    FrameSlot,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    Landmark,
    RawFrame,
    Sequence,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_features.json"

CONFIG = Config()


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    if not GOLDEN.exists():
        pytest.fail(f"falta {GOLDEN}. Generarlo con `make golden`.")
    data: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return data


def frame_from_input(payload: dict[str, Any]) -> RawFrame:
    """Reconstruye el frame **desde el JSON**, no desde el generador.

    Es deliberado: si el test partiera de los mismos objetos que usó el generador,
    verificaría que el generador es consistente consigo mismo y no que el archivo
    en disco describe lo que dice describir. TypeScript también leerá el JSON.
    """
    return RawFrame(
        landmarks=tuple(Landmark(x=x, y=y, z=z) for x, y, z in payload["landmarks"]),
        width=payload["width"],
        height=payload["height"],
        handedness=Handedness(payload["handedness"]),
        handedness_score=1.0,
        detection_score=1.0,
    )


def slot_from_input(payload: dict[str, Any]) -> FrameSlot:
    if not payload["valid"]:
        return InvalidFrame(reason=InvalidReason(payload["reason"]))
    return frame_from_input(payload)


def assert_vectors_close(
    actual: list[float] | tuple[float, ...],
    expected: list[float] | tuple[float, ...],
    tolerance: float,
    context: str,
) -> None:
    assert len(actual) == len(expected), context
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert a == pytest.approx(b, abs=tolerance), f"{context}: componente {index}"


# --------------------------------------------------------------------------- #
# Estructura del archivo
# --------------------------------------------------------------------------- #


def test_el_archivo_declara_la_version_del_spec(document: dict[str, Any]) -> None:
    assert document["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert document["segmentation_spec_version"] == SEGMENTATION_SPEC_VERSION
    assert document["tolerance"] == 1e-6


def test_las_dos_versiones_de_contrato_viven_en_modulos_distintos() -> None:
    """Se incrementan por motivos distintos: cambiar cómo se mide la velocidad no
    puede obligar a reentrenar los modelos, y para que eso siga siendo cierto las
    dos constantes no pueden acabar siendo la misma."""
    assert "SEGMENTATION_SPEC_VERSION" not in vars(lsm.features)
    assert "FEATURE_SPEC_VERSION" not in vars(lsm.segmentation)


def test_cubre_la_cobertura_minima_obligatoria(document: dict[str, Any]) -> None:
    assert len(document["cases"]) >= MINIMUM_CASES


def test_los_identificadores_son_unicos(document: dict[str, Any]) -> None:
    ids = [case["id"] for case in document["cases"]]
    sequence_ids = [case["id"] for case in document["sequence_cases"]]

    assert len(set(ids)) == len(ids)
    assert len(set(sequence_ids)) == len(sequence_ids)


def test_cada_caso_dice_que_valida(document: dict[str, Any]) -> None:
    """Un golden vector sin explicación es un número mágico con formato JSON."""
    for case in [*document["cases"], *document["sequence_cases"]]:
        assert case["description"].strip()
        assert case["validates"].strip()


# --------------------------------------------------------------------------- #
# Casos por frame
# --------------------------------------------------------------------------- #


def test_cada_caso_reproduce_sus_features(document: dict[str, Any]) -> None:
    tolerance = document["tolerance"]

    for case in document["cases"]:
        frame = frame_from_input(case["input"])
        outcome = extract_sequence_features(Sequence(frames=(frame,)), CONFIG)

        if case["expected_features"] is None:
            assert isinstance(outcome, ExtractionRejected), case["id"]
            assert str(outcome.reason) == case["expected_invalid_reason"], case["id"]
            continue

        assert isinstance(outcome, SequenceFeatures), case["id"]
        assert_vectors_close(
            outcome.frames[0].values,
            case["expected_features"],
            tolerance,
            case["id"],
        )


def test_los_pares_de_invariancia_coinciden_de_verdad(document: dict[str, Any]) -> None:
    """El test más valioso del archivo: verifica que la misma seña, hecha con la
    otra mano o desde otra distancia, produce el mismo vector."""
    tolerance = document["tolerance"]
    by_id = {case["id"]: case for case in document["cases"]}

    pairs = [case for case in document["cases"] if case.get("same_features_as")]
    assert len(pairs) >= 10, "faltan pares de invariancia en la cobertura"

    for case in pairs:
        reference = by_id[case["same_features_as"]]
        assert reference["expected_features"] is not None, case["id"]
        assert_vectors_close(
            case["expected_features"],
            reference["expected_features"],
            tolerance,
            f"{case['id']} vs {reference['id']}",
        )


def test_hay_un_caso_degenerado_con_centinela_explicito(
    document: dict[str, Any],
) -> None:
    invalid = [case for case in document["cases"] if case["expected_features"] is None]

    assert invalid
    assert all(
        case["expected_invalid_reason"] == str(InvalidReason.SCALE_TOO_SMALL)
        for case in invalid
    )


def test_el_control_negativo_no_coincide_con_el_caso_base(
    document: dict[str, Any],
) -> None:
    """Sin esto, una implementación que devuelva 42 ceros pasaría toda la tabla."""
    by_id = {case["id"]: case for case in document["cases"]}
    base = by_id[BASE_ID]["expected_features"]
    fist = by_id["fist_configuration"]["expected_features"]

    assert max(abs(a - b) for a, b in zip(base, fist, strict=True)) > 0.1


# --------------------------------------------------------------------------- #
# Casos de secuencia
# --------------------------------------------------------------------------- #


def test_cada_caso_de_secuencia_reproduce_su_salida(document: dict[str, Any]) -> None:
    tolerance = document["tolerance"]

    for case in document["sequence_cases"]:
        stream: FrameStream = tuple(
            slot_from_input(entry) for entry in case["input"]["frames"]
        )
        runs = split_valid_runs(stream)
        expected = case["expected"]

        assert len(runs) == expected["run_count"], case["id"]

        for run, expected_run in zip(runs, expected["runs"], strict=True):
            outcome = extract_sequence_features(run, CONFIG)
            assert isinstance(outcome, SequenceFeatures), case["id"]
            assert len(run) == expected_run["length"], case["id"]

            assert outcome.static.dispersion == pytest.approx(
                expected_run["dispersion"], abs=tolerance
            ), case["id"]
            assert outcome.trajectory.mean_scale == pytest.approx(
                expected_run["mean_scale"], abs=tolerance
            ), case["id"]

            for point, expected_point in zip(
                outcome.trajectory.points, expected_run["trajectory"], strict=True
            ):
                assert_vectors_close(point, expected_point, tolerance, case["id"])

            # §6: el contrato de segmentación viaja en el mismo archivo.
            assert_vectors_close(
                outcome.scales, expected_run["scales"], tolerance, case["id"]
            )
            assert_vectors_close(
                outcome.velocities, expected_run["velocities"], tolerance, case["id"]
            )

            if expected_run["dynamic_rows"] is None:
                assert isinstance(outcome.dynamic, DynamicUnavailable), case["id"]
                assert (
                    str(outcome.dynamic.reason)
                    == expected_run["dynamic_unavailable_reason"]
                )
                continue

            assert not isinstance(outcome.dynamic, DynamicUnavailable), case["id"]
            for row, expected_row in zip(
                outcome.dynamic.rows, expected_run["dynamic_rows"], strict=True
            ):
                assert_vectors_close(row, expected_row, tolerance, case["id"])


def test_las_filas_dinamicas_miden_24_por_44(document: dict[str, Any]) -> None:
    rows = [
        run["dynamic_rows"]
        for case in document["sequence_cases"]
        for run in case["expected"]["runs"]
        if run["dynamic_rows"] is not None
    ]

    assert rows
    for matrix in rows:
        assert len(matrix) == 24
        assert all(len(row) == 44 for row in matrix)


def test_la_trayectoria_ponderada_es_el_final_de_cada_fila(
    document: dict[str, Any],
) -> None:
    """§3.3: g_t = concat(f_t, w_τ · τ_t). Las dos últimas columnas son el trazo."""
    weight = document["config"]["trajectory_weight"]

    for case in document["sequence_cases"]:
        for run in case["expected"]["runs"]:
            if run["dynamic_rows"] is None:
                continue
            for row, point in zip(
                run["dynamic_rows"], run["resampled_trajectory"], strict=True
            ):
                assert row[42] == pytest.approx(weight * point[0], abs=1e-12)
                assert row[43] == pytest.approx(weight * point[1], abs=1e-12)


def test_la_secuencia_estatica_no_traza_nada(document: dict[str, Any]) -> None:
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    run = by_id["still_sequence"]["expected"]["runs"][0]

    assert all(point == [0.0, 0.0] for point in run["trajectory"])


def test_el_trazo_dinamico_si_traza(document: dict[str, Any]) -> None:
    """El par que separa una J de una I: misma forma por frame, distinto recorrido."""
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    still = by_id["still_sequence"]["expected"]["runs"][0]
    arc = by_id["trajectory_arc"]["expected"]["runs"][0]

    assert_vectors_close(
        still["frame_features"][0], arc["frame_features"][0], 1e-9, "misma forma"
    )
    assert max(abs(x) + abs(y) for x, y in arc["trajectory"]) > 0.5


def test_las_interrupciones_al_inicio_medio_y_final_se_comportan_distinto(
    document: dict[str, Any],
) -> None:
    """Dónde cae el hueco importa, no solo cuántos hay."""
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    full = by_id["trajectory_arc"]["expected"]
    start = by_id["interrupted_at_start"]["expected"]
    middle = by_id["interrupted_in_middle"]["expected"]
    end = by_id["interrupted_at_end"]["expected"]

    # En medio: dos secuencias, no una con un salto.
    assert middle["run_count"] == 2
    assert [run["length"] for run in middle["runs"]] == [5, 4]
    assert start["run_count"] == 1
    assert end["run_count"] == 1

    # Al inicio y al final se recorta un frame, pero no de la misma manera.
    assert start["runs"][0]["length"] == full["runs"][0]["length"] - 1
    assert end["runs"][0]["length"] == full["runs"][0]["length"] - 1

    # Al perder el primer frame, el origen de la trayectoria se mueve: el resto
    # del trazo se mide desde otro punto.
    assert start["runs"][0]["trajectory"][-1] != full["runs"][0]["trajectory"][-1]
    # Al perder el último, el trazo restante es idéntico al del caso completo.
    assert end["runs"][0]["trajectory"][0] == full["runs"][0]["trajectory"][0]


def test_una_secuencia_corta_no_produce_canal_dinamico(
    document: dict[str, Any],
) -> None:
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    run = by_id["too_few_source_frames"]["expected"]["runs"][0]

    assert run["dynamic_rows"] is None
    assert run["dynamic_unavailable_reason"] == "TOO_FEW_SOURCE_FRAMES"
    assert run["frame_features"], "las features estáticas sí deben estar"


def test_la_velocidad_de_una_mano_quieta_es_cero(document: dict[str, Any]) -> None:
    """§6 verificable desde TypeScript sin tener que montar la máquina de estados."""
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    run = by_id["still_sequence"]["expected"]["runs"][0]

    assert run["velocities"]
    assert all(velocity == 0.0 for velocity in run["velocities"])


def test_la_velocidad_de_un_trazo_no_es_cero(document: dict[str, Any]) -> None:
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    run = by_id["trajectory_arc"]["expected"]["runs"][0]

    assert min(run["velocities"]) > 0.0


def test_la_velocidad_no_depende_de_la_distancia_a_la_camara(
    document: dict[str, Any],
) -> None:
    """El mismo trazo ejecutado más cerca: mano al doble, recorrido al doble."""
    by_id = {case["id"]: case for case in document["sequence_cases"]}
    far = by_id["trajectory_arc"]["expected"]["runs"][0]
    near = by_id["trajectory_arc_closer"]["expected"]["runs"][0]

    assert_vectors_close(far["velocities"], near["velocities"], 1e-9, "velocidad")


def test_hay_una_velocidad_por_cada_par_de_frames(document: dict[str, Any]) -> None:
    for case in document["sequence_cases"]:
        for run in case["expected"]["runs"]:
            assert len(run["scales"]) == run["length"], case["id"]
            assert len(run["velocities"]) == run["length"] - 1, case["id"]
