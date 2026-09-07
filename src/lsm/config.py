"""Configuración validada del traductor.

`CLAUDE.md` §5: cero umbrales hardcodeados. Velocidades, ventanas, confianzas
mínimas y cooldowns viven aquí y se validan al cargar, no a mitad de una demo.

**Qué NO va en este archivo.** Las constantes del contrato de features no son
umbrales ajustables: la dimensión 42, `T_ref = 24`, `MIN_SCALE = 1e-6` y
`FEATURE_SPEC_VERSION` definen el formato que la implementación de TypeScript
tiene que reproducir bit a bit. Si fueran configurables, dos instalaciones con
distinto `config.yaml` producirían vectores incompatibles sin que nada lo
detectara. Viven en `lsm.features` como constantes de módulo y cambiarlas obliga a
incrementar `FEATURE_SPEC_VERSION`. Ver `docs/adr/0002-formato-de-features.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    """Base de todas las secciones: inmutable y sin campos desconocidos.

    `extra="forbid"` es deliberado: un typo en `config.yaml` debe reventar al
    cargar y no caer en silencio al valor por defecto, que es exactamente la clase
    de error que se descubre tres semanas después mirando métricas raras.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class QualityConfig(_Section):
    """Criterios de calidad de una ventana (`feature-spec.md` §2)."""

    #: Umbral de σ, la dispersión media por componente de la ventana. Por encima,
    #: la ventana se considera inestable y se rechaza antes de clasificar.
    #: Unidades: las del vector de features, o sea unidades de mano.
    max_dispersion: float = Field(default=0.08, gt=0.0, le=10.0)


class FeaturesConfig(_Section):
    """Parámetros ajustables de la extracción (`feature-spec.md` §3.3)."""

    #: w_τ: peso del canal de trayectoria frente al de forma. La forma aporta 42
    #: componentes y la trayectoria 2; sin ponderar, el trazo —que es lo único que
    #: distingue una J de una I— queda invisible en la distancia euclidiana.
    trajectory_weight: float = Field(default=4.0, ge=0.0, le=100.0)


class SmoothingConfig(_Section):
    """Media móvil exponencial sobre landmarks crudos (`feature-spec.md` §4)."""

    #: α = 1.0 desactiva el suavizado. Valores menores reducen el jitter de
    #: MediaPipe a costa de latencia y de emborronar los movimientos rápidos, lo
    #: que perjudica justo a las señas dinámicas. Si se activa aquí, debe
    #: activarse idénticamente en la implementación web.
    alpha: float = Field(default=1.0, gt=0.0, le=1.0)


class DtwConfig(_Section):
    """Parámetros del clasificador dinámico (`feature-spec.md` §3.2 y §3.4)."""

    #: Radio de la ventana de Sakoe-Chiba. Un radio ≥ T_ref equivale a no imponer
    #: banda; no es un error, solo un DTW más caro.
    band_radius: int = Field(default=6, ge=1, le=1000)

    #: Mínimo de frames válidos de origen para remuestrear a T_ref. Por debajo, la
    #: secuencia se rechaza en vez de interpolarse: estirar 2 frames a 24 inventa
    #: una trayectoria que nadie ejecutó.
    min_source_frames: int = Field(default=4, ge=1, le=1000)


class SegmentationConfig(_Section):
    """Umbrales de la máquina de estados (`ARQUITECTURA.md` §4.2)."""

    #: Tamaño del buffer circular de frames recientes, en frames.
    buffer_size: int = Field(default=24, ge=2, le=300)

    #: Por debajo de esto, el frame del detector se marca inválido.
    min_detection_score: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Velocidad por debajo de la cual se considera que la mano está quieta.
    #: Unidades de mano por frame; definida en `docs/feature-spec.md` §6.
    velocity_threshold: float = Field(default=0.02, gt=0.0, le=100.0)

    #: Cuántos frames consecutivos por debajo del umbral hacen falta para pasar de
    #: TRACKING a STABLE.
    stable_frames: int = Field(default=5, ge=1, le=300)

    #: Confianza mínima para emitir una letra. Por debajo se emite UNKNOWN y no se
    #: agrega nada al texto (`ARQUITECTURA.md` §4.4).
    min_confidence: float = Field(default=0.6, gt=0.0, le=1.0)

    #: Cooldown tras emitir una letra, para no repetirla mientras la mano sigue
    #: quieta.
    emit_cooldown_frames: int = Field(default=12, ge=1, le=300)

    #: Cooldown tras un rechazo (confianza insuficiente o ventana inestable).
    #: Más corto que el de emisión a propósito: ver `lsm.segmentation`.
    reject_cooldown_frames: int = Field(default=4, ge=1, le=300)

    #: Frames consecutivos sin mano que llevan de vuelta a IDLE.
    missing_frames_to_idle: int = Field(default=8, ge=1, le=300)

    @model_validator(mode="after")
    def _coherencia_entre_umbrales(self) -> SegmentationConfig:
        if self.stable_frames > self.buffer_size:
            msg = (
                f"stable_frames ({self.stable_frames}) excede buffer_size "
                f"({self.buffer_size}): nunca se alcanzaría STABLE"
            )
            raise ValueError(msg)
        if self.reject_cooldown_frames > self.emit_cooldown_frames:
            msg = (
                f"reject_cooldown_frames ({self.reject_cooldown_frames}) excede "
                f"emit_cooldown_frames ({self.emit_cooldown_frames}): un rechazo "
                "costaría más que un acierto y una letra apenas bajo el umbral "
                "obligaría a rehacer la seña entera"
            )
            raise ValueError(msg)
        return self


class Config(_Section):
    """Configuración completa. Inmutable y validada."""

    quality: QualityConfig = Field(default_factory=QualityConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    dtw: DtwConfig = Field(default_factory=DtwConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)


def load_config(path: Path | str) -> Config:
    """Lee y valida un `config.yaml`.

    Es la única función de este módulo que toca disco. El núcleo puro recibe un
    `Config` ya construido, nunca una ruta.
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    parsed: Any = yaml.safe_load(raw_text)
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        msg = (
            f"{path}: se esperaba un mapeo en la raíz, se leyó {type(parsed).__name__}"
        )
        raise ValueError(msg)
    return Config.model_validate(parsed)
