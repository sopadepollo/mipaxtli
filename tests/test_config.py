"""La configuración es el único lugar donde viven los umbrales.

`CLAUDE.md` §5: cero umbrales hardcodeados. Estos tests fijan que el esquema
cubra todo lo que la máquina de estados y la tubería de features necesitan, y
que los rangos inválidos se rechacen al cargar y no a las tres horas de demo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lsm.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_por_defecto_cubre_los_umbrales_del_contrato() -> None:
    config = Config()

    assert config.quality.max_dispersion > 0.0
    assert config.features.trajectory_weight == 1.0
    assert config.smoothing.enabled is True
    assert config.dtw.band_radius == 6
    assert config.dtw.min_source_frames >= 1

    segmentation = config.segmentation
    assert segmentation.buffer_ms > 0
    assert segmentation.velocity_threshold_per_s > 0.0
    assert segmentation.stable_ms > 0
    assert segmentation.min_confidence > 0.0
    assert segmentation.emit_cooldown_ms > 0
    assert segmentation.reject_cooldown_ms > 0
    assert segmentation.missing_to_idle_ms > 0
    assert segmentation.min_detection_score > 0.0


def test_la_configuracion_es_inmutable() -> None:
    config = Config()

    with pytest.raises(ValidationError):
        config.smoothing.min_cutoff = 0.5


def test_alpha_de_suavizado_solo_admite_el_intervalo_semiabierto() -> None:
    Config.model_validate({"smoothing": {"min_cutoff": 1.0, "beta": 0.0}})
    Config.model_validate({"smoothing": {"enabled": True, "beta": 0.3}})

    with pytest.raises(ValidationError):
        Config.model_validate({"smoothing": {"min_cutoff": 0.0}})
    with pytest.raises(ValidationError):
        Config.model_validate({"smoothing": {"beta": -1.0}})


def test_confianza_minima_fuera_de_rango_se_rechaza() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"segmentation": {"min_confidence": 1.4}})


def test_campos_desconocidos_se_rechazan() -> None:
    """Un typo en config.yaml no debe caer silenciosamente al valor por defecto."""
    with pytest.raises(ValidationError):
        Config.model_validate({"smoothing": {"alfa": 0.5}})


def test_el_cooldown_de_rechazo_no_puede_superar_al_de_emision() -> None:
    """Ver la decisión documentada en `segmentation.py`.

    Un rechazo (UNKNOWN) debe costar menos que una emisión: si costara más, una
    letra legítima que quedó apenas bajo el umbral obligaría a rehacer la seña.
    """
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"segmentation": {"emit_cooldown_ms": 4, "reject_cooldown_ms": 9}}
        )


def test_los_frames_estables_no_pueden_exceder_el_buffer() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"segmentation": {"buffer_ms": 8, "stable_ms": 9}})


def test_el_config_yaml_de_ejemplo_es_valido_y_coincide_con_los_defaults() -> None:
    config = load_config(REPO_ROOT / "config.yaml")

    assert config == Config()


def test_el_espacio_exige_mas_ausencia_que_la_vuelta_a_idle() -> None:
    """Si bastara con lo que la maquina de estados considera "mano perdida", un
    parpadeo del detector escribiria un espacio. El espacio es una intencion de
    quien firma, no un fallo de deteccion."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "segmentation": {"missing_to_idle_ms": 8},
                "spelling": {"space_after_absent_ms": 8},
            }
        )


def test_el_espacio_por_defecto_es_un_segundo_de_verdad() -> None:
    """Un segundo, y ahora lo es a cualquier tasa.

    Antes eran 30 cuadros «que son un segundo a 30 fps», y en la maquina medida
    —17.8 fps— eran 1685 ms. Ver `docs/adr/0013-la-ventana-mezclada.md`.
    """
    assert Config().spelling.space_after_absent_ms == 1000.0


def test_la_telemetria_tambien_tiene_su_seccion() -> None:
    """`CLAUDE.md` §5 no hace excepciones: la ventana sobre la que se promedia
    el fps que se ve en pantalla y la duracion de la medicion son umbrales, y
    viven en `config.yaml` como los demas."""
    telemetry = Config().telemetry

    assert telemetry.fps_window_frames > 0
    assert telemetry.benchmark_seconds > 0.0


def test_una_ventana_de_fps_vacia_se_rechaza() -> None:
    """Promediar sobre cero cuadros no significa nada, y el HUD no deberia
    tener que defenderse de una configuracion imposible."""
    with pytest.raises(ValidationError):
        Config.model_validate({"telemetry": {"fps_window_frames": 0}})


def test_una_medicion_de_duracion_cero_se_rechaza() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"telemetry": {"benchmark_seconds": 0.0}})


def test_texto_a_senas_tiene_su_seccion() -> None:
    """Duraciones, fps del render y límites de velocidad son umbrales: viven en
    `config.yaml` como los demás (`CLAUDE.md` regla 5)."""
    signs = Config().signs

    assert signs.static_hold_ms > 0.0
    assert signs.dynamic_loops >= 1
    assert signs.word_gap_ms > 0.0
    assert signs.render_fps >= 1
    assert signs.canvas_px >= 64
    assert 0.0 <= signs.canvas_margin < 0.5
    assert signs.speed_min < 1.0 <= signs.speed_max
    assert signs.tick_ms >= 1


def test_un_rango_de_velocidad_vacio_se_rechaza() -> None:
    """`Faster`/`Slower` acotan entre `speed_min` y `speed_max`; con el rango al
    revés no hay velocidad válida y el reproductor no debería descubrirlo."""
    with pytest.raises(ValidationError):
        Config.model_validate({"signs": {"speed_min": 2.0, "speed_max": 1.0}})


# --------------------------------------------------------------------------- #
# Camino dinámico (ADR 0015)
# --------------------------------------------------------------------------- #


def test_motion_threshold_no_puede_quedar_bajo_velocity_threshold() -> None:
    """Si quedara por debajo, un mismo frame contaría como quietud para STABLE y
    como movimiento para el candidato: los dos caminos dejarían de excluirse."""
    Config.model_validate(
        {
            "segmentation": {
                "velocity_threshold_per_s": 0.6,
                "motion_threshold_per_s": 0.6,
            }
        }
    )

    with pytest.raises(ValidationError, match="excluyentes"):
        Config.model_validate(
            {
                "segmentation": {
                    "velocity_threshold_per_s": 0.9,
                    "motion_threshold_per_s": 0.75,
                }
            }
        )


def test_el_reposo_que_cierra_un_trazo_no_es_mas_corto_que_stable() -> None:
    """Una pausa que el camino estático ya llama quietud partiría la dinámica."""
    with pytest.raises(ValidationError, match="partiría una dinámica"):
        Config.model_validate(
            {"segmentation": {"stable_ms": 200.0, "motion_confirm_low_ms": 150.0}}
        )


def test_motion_min_tiene_que_ser_menor_que_motion_max() -> None:
    with pytest.raises(ValidationError, match="motion_max_ms"):
        Config.model_validate(
            {"segmentation": {"motion_min_ms": 500.0, "motion_max_ms": 500.0}}
        )


def test_config_yaml_trae_los_umbrales_del_camino_dinamico() -> None:
    """Explícitos en el archivo, no heredados del valor por defecto: son
    medidos (ADR 0015) y quien lea `config.yaml` tiene que verlos."""
    texto = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    config = load_config(REPO_ROOT / "config.yaml")

    for campo in (
        "motion_threshold_per_s",
        "motion_min_ms",
        "motion_confirm_low_ms",
        "motion_max_ms",
        "max_distance",
    ):
        assert f"{campo}:" in texto
    assert config.segmentation.motion_threshold_per_s >= (
        config.segmentation.velocity_threshold_per_s
    )
