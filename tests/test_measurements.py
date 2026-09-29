"""Las mediciones del Paso 1 (ADR 0026): lo que decide cómo se lee cada número."""

from __future__ import annotations

from lsm.landmark_stats import bone_lengths, canonical_points, mcp_dorsal_elevation
from lsm.measurements import Phase, attempt_bounds, phases_from_states
from lsm.synthetic import canonical_hand, to_frame
from lsm.types import Points3, RawFrame


def test_los_intentos_son_tramos_contiguos_de_la_misma_repeticion() -> None:
    prompts = [None, "X", "X", "X", "X", None, "Q"]
    reps = [None, 1, 1, 2, 2, None, 1]

    assert attempt_bounds(prompts, reps) == ((1, 3), (3, 5), (6, 7))


def test_la_fase_del_trazo_empieza_antes_de_entrar_al_candidato() -> None:
    estados = ["TRACKING"] * 4 + ["DYNAMIC_CANDIDATE"] * 3 + ["EMIT"] * 2
    reloj = [100.0 * i for i in range(9)]

    fases = phases_from_states(estados, reloj, [(0, 9)], pre_candidate_ms=250.0)

    assert fases == (
        Phase.ANTES,
        Phase.ANTES,
        Phase.DURANTE,  # 200 ms ≥ 400 − 250
        Phase.DURANTE,
        Phase.DURANTE,
        Phase.DURANTE,
        Phase.DURANTE,
        Phase.FINAL,
        Phase.FINAL,
    )


def test_un_intento_sin_candidato_no_tiene_trazo() -> None:
    fases = phases_from_states(["TRACKING"] * 3, [0.0, 1.0, 2.0], [(0, 3)], 300.0)

    assert fases == (Phase.SIN_TRAZO,) * 3


def _frame(points: Points3) -> RawFrame:
    return to_frame(points, width=1280, height=720)


def test_una_mano_plana_no_levanta_ninguna_falange() -> None:
    """La mano sintética abierta está en un plano: elevación dorsal nula. El
    signo con manos reales (un puño, negativo) está medido en el ADR 0026."""
    abierta = mcp_dorsal_elevation(canonical_points(_frame(canonical_hand())))

    assert abierta is not None
    assert all(abs(a) < 1.0 for a in abierta)


def test_los_huesos_se_miden_en_palmas_y_no_dependen_de_la_distancia() -> None:
    from lsm.synthetic import scaled

    cerca = bone_lengths(canonical_points(_frame(canonical_hand())), depth=True)
    lejos = bone_lengths(
        canonical_points(_frame(scaled(canonical_hand(), 0.5))), depth=True
    )

    assert cerca is not None
    assert lejos is not None
    assert all(abs(a - b) < 1e-9 for a, b in zip(cerca, lejos, strict=True))
