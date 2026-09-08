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
    assert segmentation.buffer_size > 0
    assert segmentation.velocity_threshold > 0.0
    assert segmentation.stable_frames > 0
    assert segmentation.min_confidence > 0.0
    assert segmentation.emit_cooldown_frames > 0
    assert segmentation.reject_cooldown_frames > 0
    assert segmentation.missing_frames_to_idle > 0
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
            {"segmentation": {"emit_cooldown_frames": 4, "reject_cooldown_frames": 9}}
        )


def test_los_frames_estables_no_pueden_exceder_el_buffer() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"segmentation": {"buffer_size": 8, "stable_frames": 9}})


def test_el_config_yaml_de_ejemplo_es_valido_y_coincide_con_los_defaults() -> None:
    config = load_config(REPO_ROOT / "config.yaml")

    assert config == Config()
