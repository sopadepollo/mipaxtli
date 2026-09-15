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
    assert config.features.trajectory_weight == 4.0
    assert config.smoothing.alpha == 1.0
    assert config.dtw.band_radius == 6
    assert config.dtw.min_source_frames >= 1

    segmentation = config.segmentation
    assert segmentation.buffer_ms > 0
    assert segmentation.velocity_threshold > 0.0
    assert segmentation.stable_ms > 0
    assert segmentation.min_confidence > 0.0
    assert segmentation.emit_cooldown_ms > 0
    assert segmentation.reject_cooldown_ms > 0
    assert segmentation.missing_to_idle_ms > 0
    assert segmentation.min_detection_score > 0.0


def test_la_configuracion_es_inmutable() -> None:
    config = Config()

    with pytest.raises(ValidationError):
        config.smoothing.alpha = 0.5


def test_alpha_de_suavizado_solo_admite_el_intervalo_semiabierto() -> None:
    Config.model_validate({"smoothing": {"alpha": 1.0}})
    Config.model_validate({"smoothing": {"alpha": 0.3}})

    with pytest.raises(ValidationError):
        Config.model_validate({"smoothing": {"alpha": 0.0}})
    with pytest.raises(ValidationError):
        Config.model_validate({"smoothing": {"alpha": 1.5}})


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
