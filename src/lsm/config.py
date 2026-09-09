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

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metric(StrEnum):
    """Distancia con la que `static_knn` compara contra los centroides.

    Vive aquí y no en `classifiers/static_knn.py` porque es un valor de
    `config.yaml` y Pydantic tiene que validarlo al cargar: un `metric: euclidian`
    mal escrito debe reventar en el arranque, no elegir una distancia por defecto
    en silencio. El clasificador la reexporta para quien solo lo importe a él.
    """

    #: Norma L2 sobre los 42 componentes. Sensible al tamaño del vector.
    EUCLIDEAN = "euclidean"
    #: `1 − cos(u, v)`. Ignora la magnitud y mira solo la dirección.
    COSINE = "cosine"


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


class StaticKnnConfig(_Section):
    """Clasificador de letras estáticas por centroides (`ARQUITECTURA.md` §4.3).

    Los tres campos viajan **dentro del modelo exportado**, no solo en este
    archivo: la reimplementación de JavaScript tiene que poder reproducir la
    decisión completa, y una que acertara la etiqueta pero no supiera cuándo
    callarse no serviría de nada (`ARQUITECTURA.md` §4.4 y §4.6).
    """

    #: Distancia entre el vector de la ventana y cada centroide.
    #: Medido con el barrido de la Fase 2 contra el dataset `fe0ada8710b6`:
    #: la coseno gana por +9 puntos. Ver `docs/adr/0011-calibracion-de-la-fase-2.md`.
    metric: Metric = Metric.COSINE

    #: Distancia máxima al centroide más cercano para no devolver `UNKNOWN`. Es
    #: la puerta que tapa lo que no se parece a ninguna letra: una mano saludando
    #: cae lejos de las 22 clases, pero *alguna* será la menos lejana.
    #:
    #: Unidades: las del vector de features, o sea unidades de mano. **Valor de
    #: partida razonado, no medido**; se calibra con el barrido de `lsm-eval`.
    #: En unidades de la métrica activa. Con coseno la escala es `[0, 2]`, así
    #: que este 0.25 no es comparable con el 1.0 que había para la euclidiana.
    max_distance: float = Field(default=0.25, gt=0.0, le=1000.0)

    #: Confianza mínima para no devolver `UNKNOWN`, en la escala del margen
    #: normalizado `d₂ / (d₁ + d₂)`: vale 0.5 con dos centroides empatados y
    #: tiende a 1 cuando el más cercano gana con holgura.
    #:
    #: Es la puerta que tapa las confundibles del §4.8. Un 0.5 la desactiva.
    #: Comparte escala con `segmentation.min_confidence` a propósito, para que las
    #: dos se puedan leer juntas; esta es el piso del clasificador y aquella es el
    #: piso, más alto, que la máquina de estados exige para escribir una letra.
    #: 0.6 se eligió **en contra** del barrido, que recomienda 0.5. Con 0.5 la
    #: puerta queda apagada —la confianza vive en `[0.5, 1]`— y eso gana en la
    #: evaluación offline solo porque ahí toda ventana es una letra de verdad.
    #: En video en vivo la mayoría de los frames no lo son. Ver el ADR 0011.
    min_margin: float = Field(default=0.6, ge=0.5, le=1.0)


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


class HandsConfig(_Section):
    """Parámetros del detector de manos (`src/lsm/io/hands.py`).

    Son los knobs que MediaPipe consume **antes** de que exista un `RawFrame`, y
    por eso no viven en `segmentation`: la máquina de estados decide qué hacer con
    los frames que ya salieron del detector, no cómo detectarlos.
    """

    #: Ruta del bundle `hand_landmarker.task`. No se versiona en el repositorio
    #: (son ~8 MB); se descarga con `make model`. Ver `docs/dataset-schema.md`.
    model_path: Path = Field(default=Path("data/models/hand_landmarker.task"))

    #: Cuántas manos busca el detector. **No es 1 a propósito.** `feature-spec.md`
    #: §0.3 manda quedarse con la de mayor score cuando hay varias, y para poder
    #: elegir hay que verlas: con `num_hands = 1` MediaPipe devuelve la que él
    #: prefiera y la regla del contrato se vuelve inaplicable.
    num_hands: int = Field(default=2, ge=1, le=4)

    #: Umbral interno de detección de palma. Es el mismo concepto que
    #: `segmentation.min_detection_score`, pero aplicado dentro de MediaPipe: lo
    #: que no lo supera ni siquiera llega a ser un candidato.
    min_hand_detection_confidence: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Umbral de presencia de la mano entre frames de video.
    min_hand_presence_confidence: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Umbral del rastreador temporal. Por debajo, MediaPipe vuelve a detectar
    #: desde cero en vez de seguir la mano del frame anterior.
    min_tracking_confidence: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Si hay que **invertir** la lateralidad que reporta el detector.
    #:
    #: **Medido el 2026-09-08 con MediaPipe 1.0.1: no hay que invertirla.**
    #: Alimentado con el cuadro sin espejar —como obliga `feature-spec.md` §0.3—
    #: devuelve la mano anatómica: mano derecha real, `"Right"`. De ahí el `false`.
    #:
    #: La documentación histórica de MediaPipe afirmaba lo contrario ("handedness
    #: is determined assuming the input image is mirrored"), y ese fue el valor por
    #: defecto hasta comprobarlo contra una cámara. Se conserva el interruptor
    #: porque la convención ya cambió una vez entre versiones y puede volver a
    #: cambiar.
    #:
    #: Equivocarse aquí no rompe nada visible: el paso 2 canoniza **todas** las
    #: muestras hacia la mano contraria, el vector queda coherente consigo mismo y
    #: el modelo entrena sin quejarse. Por eso no se decide leyendo, se decide
    #: mirando: `lsm-capture calibrar` lo verifica y lo deja registrado, y `grabar`
    #: no arranca sin ese registro.
    mediapipe_reports_mirrored_handedness: bool = Field(default=False)


class SpellingConfig(_Section):
    """Acumulación de letras en palabras (`ARQUITECTURA.md` §4.2)."""

    #: Frames consecutivos sin mano antes de cerrar la palabra en curso. A 30 fps,
    #: 30 frames es un segundo. Es el único gesto de control del proyecto: bajar
    #: la mano entre palabras es lo que se hace de todos modos, y el clasificador
    #: no tiene clases libres para un gesto dedicado.
    space_after_absent_frames: int = Field(default=30, ge=1, le=600)


class CaptureConfig(_Section):
    """Recolección de dataset (`src/lsm/cli/capture.py`, `ARQUITECTURA.md` §4.7)."""

    #: Índice de la cámara para OpenCV.
    camera_index: int = Field(default=0, ge=0, le=64)

    #: Resolución pedida a la cámara. Es una *petición*: si el driver no la
    #: soporta entrega otra, y por eso `width`/`height` viajan en cada frame en
    #: vez de darse por sabidos. El paso 1 de `feature-spec.md` los necesita.
    frame_width: int = Field(default=1280, ge=160, le=7680)
    frame_height: int = Field(default=720, ge=120, le=4320)

    #: Cuadros por segundo pedidos a la cámara. Además fija el paso de los
    #: timestamps que consume el modo VIDEO de MediaPipe.
    camera_fps: int = Field(default=30, ge=1, le=240)

    #: Espejar el preview. **Solo afecta a lo que se dibuja en pantalla**: el
    #: frame que recibe el detector nunca va espejado (`feature-spec.md` §0.3).
    preview_mirror: bool = Field(default=True)

    #: Frames que componen una muestra estática. A 30 fps, 24 frames son 800 ms
    #: de mano sostenida, que es lo que se le pide a quien graba.
    static_frames: int = Field(default=24, ge=2, le=300)

    #: Límites de una muestra dinámica. El mínimo evita guardar un trazo cortado;
    #: el máximo impide que una grabación olvidada crezca sin fin.
    dynamic_min_frames: int = Field(default=12, ge=2, le=300)
    dynamic_max_frames: int = Field(default=90, ge=2, le=1000)

    #: Longitud de arco mínima de la trayectoria τ (`feature-spec.md` §3.1) para
    #: aceptar una muestra **dinámica**, en unidades de mano.
    #:
    #: Es el análogo de `quality.max_dispersion` por el otro lado: σ rechaza a la
    #: estática que se movió, esto rechaza a la dinámica que no se movió. Sin él,
    #: una "J" en la que la mano apenas se desplazó entra al dataset y es
    #: indistinguible de una "I" — y envenena al clasificador dinámico por el lado
    #: contrario al que tapa la clase negativa.
    #:
    #: Se mide sobre τ y no sobre los píxeles porque τ ya está dividida por la
    #: escala de la mano: el mismo trazo cuenta igual de cerca que de lejos.
    #:
    #: **0.8 es un punto de partida razonado, no medido**, y se calibra en la Fase 2
    #: con trazos reales. El razonamiento: el gancho de una J recorre del orden de
    #: una o dos veces la distancia muñeca-nudillo, mientras que una mano sostenida
    #: acumula solo el temblor del detector.
    min_trajectory_arc: float = Field(default=0.8, gt=0.0, le=1000.0)

    #: Meta de repeticiones por letra y sesión. Solo alimenta el contador en
    #: pantalla: nada se bloquea al alcanzarla.
    target_samples_per_label: int = Field(default=20, ge=1, le=1000)

    @model_validator(mode="after")
    def _coherencia_de_la_captura(self) -> CaptureConfig:
        if self.dynamic_min_frames > self.dynamic_max_frames:
            msg = (
                f"dynamic_min_frames ({self.dynamic_min_frames}) excede "
                f"dynamic_max_frames ({self.dynamic_max_frames}): ninguna "
                "grabación dinámica podría aceptarse"
            )
            raise ValueError(msg)
        return self


class Config(_Section):
    """Configuración completa. Inmutable y validada."""

    quality: QualityConfig = Field(default_factory=QualityConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    static_knn: StaticKnnConfig = Field(default_factory=StaticKnnConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    dtw: DtwConfig = Field(default_factory=DtwConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    hands: HandsConfig = Field(default_factory=HandsConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    spelling: SpellingConfig = Field(default_factory=SpellingConfig)

    @model_validator(mode="after")
    def _la_captura_alcanza_para_el_canal_dinamico(self) -> Config:
        """Una muestra dinámica tiene que poder remuestrearse a `T_ref`.

        `feature-spec.md` §3.2 rechaza las secuencias con menos de
        `dtw.min_source_frames` frames en vez de interpolarlas. Si la captura
        aceptara muestras por debajo de ese mínimo, el dataset se llenaría de
        grabaciones que el clasificador dinámico no puede leer, y el síntoma
        aparecería en la Fase 5 con la gente ya grabada y en su casa.
        """
        if self.capture.dynamic_min_frames < self.dtw.min_source_frames:
            msg = (
                f"capture.dynamic_min_frames ({self.capture.dynamic_min_frames}) "
                f"es menor que dtw.min_source_frames ({self.dtw.min_source_frames}): "
                "se grabarían muestras dinámicas que el remuestreo del §3.2 rechaza"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _el_espacio_no_lo_dispara_un_parpadeo(self) -> Config:
        """El espacio tiene que costar más ausencia que volver a IDLE.

        `missing_frames_to_idle` es cuánto tarda la máquina de estados en dar la
        mano por perdida, y se cruza con cualquier oclusión momentánea. Si el
        espacio se disparara ahí, un parpadeo del detector partiría una palabra
        en dos y quien firma no tendría forma de evitarlo.
        """
        espacio = self.spelling.space_after_absent_frames
        idle = self.segmentation.missing_frames_to_idle
        if espacio <= idle:
            msg = (
                f"spelling.space_after_absent_frames ({espacio}) no supera "
                f"segmentation.missing_frames_to_idle ({idle}): un parpadeo "
                "del detector escribiría un espacio"
            )
            raise ValueError(msg)
        return self


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
