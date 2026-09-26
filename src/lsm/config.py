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
    #:
    #: MEDIDO en la Fase 5 (ADR 0016): 1.0 y no el 4.0 de partida. La I y la J
    #: nunca compiten dentro del DTW —llegan por caminos distintos de la
    #: segmentación—, y entre las dinámicas la forma es la señal más estable
    #: entre personas. Los golden vectors se generan con este valor y lo
    #: declaran en su bloque `config`.
    trajectory_weight: float = Field(default=1.0, ge=0.0, le=100.0)


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

    #: Costo DTW normalizado máximo contra la plantilla más cercana para no
    #: devolver `UNKNOWN`, en unidades de `g_t` (§3.3) por paso de alineación.
    #:
    #: Es la puerta que tapa lo que entra al camino dinámico sin ser una letra
    #: dinámica: un tránsito largo entre dos letras, un saludo, una estática que
    #: tembló lo bastante para cruzar el umbral de movimiento. DTW siempre tiene
    #: una plantilla más cercana, y sin esta puerta esa gana. Depende de
    #: `features.trajectory_weight`, que escala dos de las 44 componentes: si se
    #: toca una, hay que volver a medir la otra.
    #:
    #: MEDIDO (ADR 0016) con `w_τ = 1.0`: el p99 de `d₁` de los aciertos
    #: leave-one-signer-out. Puerta gruesa a propósito: la distancia no separa
    #: estáticas de dinámicas, y lo que filtra es la segmentación más el margen.
    max_distance: float = Field(default=6.0, gt=0.0, le=1000.0)


class SegmentationConfig(_Section):
    """Umbrales de la máquina de estados (`ARQUITECTURA.md` §4.2)."""

    #: Cuánto pasado recuerda el buffer circular, en **milisegundos**.
    #:
    #: Los umbrales temporales están en milisegundos y no en frames desde que se
    #: midió la tasa real (`docs/adr/0013-la-ventana-mezclada.md`). En frames, el
    #: comportamiento cambiaba con la máquina sin que nadie lo notara: estos 24
    #: frames eran los 800 ms que decía el comentario a 30 fps, y 1348 ms a los
    #: 17.8 fps que de verdad sostiene la máquina de referencia.
    #:
    #: La conversión a frames la hace `FrameThresholds.from_config` con la tasa
    #: medida en tiempo de ejecución.
    buffer_ms: float = Field(default=800.0, gt=0.0, le=60000.0)

    #: Por debajo de esto, el frame del detector se marca inválido.
    min_detection_score: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Velocidad por debajo de la cual se considera que la mano está quieta.
    #: Unidades de mano **por frame**; definida en `docs/feature-spec.md` §6.
    #:
    #: **Sigue en unidades por frame y por tanto sigue dependiendo de la tasa**,
    #: al revés que el resto de esta sección. Es una deuda conocida y anotada: a
    #: menor tasa, dos frames consecutivos están más separados en el tiempo, así
    #: que la misma mano física da un `v_t` mayor y cuesta más declararla quieta.
    #: Expresarla por segundo es el arreglo, y no se hizo aquí porque
    #: `segmentation.velocity_threshold` es un **eje del barrido de calibración de
    #: la Fase 2** (`src/lsm/cli/evaluate.py`): cambiarle la unidad invalida esa
    #: calibración y pide su propia medición. Ver el ADR 0013.
    velocity_threshold: float = Field(default=0.02, gt=0.0, le=100.0)

    #: Cuánta quietud continuada hace falta para pasar de TRACKING a STABLE, en
    #: milisegundos. Es además el **piso** de la ventana que se clasifica.
    stable_ms: float = Field(default=167.0, gt=0.0, le=60000.0)

    #: Confianza mínima para emitir una letra. Por debajo se emite UNKNOWN y no se
    #: agrega nada al texto (`ARQUITECTURA.md` §4.4). Es el **piso**: con la
    #: emisión progresiva, una confianza entre este valor y `high_confidence` no
    #: emite todavía, sigue acumulando evidencia.
    min_confidence: float = Field(default=0.6, gt=0.0, le=1.0)

    #: Confianza que emite **sin esperar más evidencia**. Entre `min_confidence`
    #: y este valor, la máquina sigue clasificando frame a frame mientras la
    #: ventana crece, y emite al agotarla.
    #:
    #: MEDIDO bajo leave-one-signer-out sobre el dataset de la Fase 2 (2437
    #: muestras no rechazadas): en 0.82 el 75.1% de las muestras emitirían de
    #: inmediato con precisión 0.9995 —un único error, `E→C` con 0.9316, que
    #: ningún umbral por debajo de 0.94 excluye—. Es la rodilla: en 0.80 pasan 4
    #: errores y en 0.85 la precisión ya no mejora (0.9994) mientras la emisión
    #: inmediata cae al 67.1%, o sea un tercio de las letras esperando la
    #: ventana entera.
    #:
    #: **Lo que la medida no cubre:** se midió sobre muestras completas de 24
    #: frames, no sobre las ventanas cortas de la emisión progresiva, que son más
    #: ruidosas. Una ventana de 5 frames alcanzará 0.82 menos veces que una de
    #: 24, así que el efecto real es acumular más de lo que predice esta tabla.
    #: El error va hacia tardar, no hacia escribir mal.
    high_confidence: float = Field(default=0.82, gt=0.0, le=1.0)

    #: Cooldown tras emitir una letra, en milisegundos, para no repetirla
    #: mientras la mano sigue quieta.
    emit_cooldown_ms: float = Field(default=400.0, gt=0.0, le=60000.0)

    #: Cooldown tras un rechazo (confianza insuficiente o ventana inestable), en
    #: milisegundos. Más corto que el de emisión a propósito: ver
    #: `lsm.segmentation`.
    reject_cooldown_ms: float = Field(default=133.0, gt=0.0, le=60000.0)

    #: Ausencia de mano continuada que lleva de vuelta a IDLE, en milisegundos.
    missing_to_idle_ms: float = Field(default=267.0, gt=0.0, le=60000.0)

    # -- Camino dinámico (feature-spec.md §6.7, ADR 0015) ------------------- #

    #: Velocidad `v_t` del §6.1 por encima de la cual un frame **cuenta como
    #: movimiento** para el camino dinámico. Es la misma métrica que
    #: `velocity_threshold`, no una segunda: lo que cambia es el umbral.
    #:
    #: Unidades de mano **por frame**, con la misma deuda anotada que
    #: `velocity_threshold`: depende de la tasa. Tiene que ser ≥ que
    #: `velocity_threshold` —se valida— para que ningún frame pueda contar a la
    #: vez como quietud para STABLE y como movimiento para el candidato.
    motion_threshold: float = Field(default=0.025, gt=0.0, le=100.0)

    #: Frames de movimiento que tiene que acumular una racha para pasar de
    #: TRACKING a DYNAMIC_CANDIDATE, en milisegundos. Cuenta frames móviles,
    #: no consecutivos: la cámara entrega cuadros duplicados y `v_t` alterna
    #: alto/casi cero en pleno trazo (ADR 0015), así que exigir consecutivos no
    #: dispararía nunca. Lo que corta una racha es `motion_confirm_low_ms`.
    motion_min_ms: float = Field(default=333.0, gt=0.0, le=60000.0)

    #: Reposo continuado —frames con `v_t < motion_threshold`— que da por
    #: terminado el movimiento, en milisegundos. En DYNAMIC_CANDIDATE es lo que
    #: dispara DYNAMIC_EMIT; en TRACKING es lo que mata una racha que no llegó a
    #: candidato. Tiene que ser más largo que el freno de un cambio de dirección
    #: de la Z o del gancho de la J, o esas letras se parten por la mitad.
    motion_confirm_low_ms: float = Field(default=400.0, gt=0.0, le=60000.0)

    #: Duración máxima de un candidato dinámico, en milisegundos, contada desde
    #: el primer frame del trazo. Por encima se descarta sin clasificar y se
    #: vuelve a TRACKING: nadie tarda eso en trazar una letra, y lo que sí dura
    #: tanto —alguien gesticulando— no debe llegar al clasificador dinámico.
    motion_max_ms: float = Field(default=4000.0, gt=0.0, le=60000.0)

    @model_validator(mode="after")
    def _coherencia_del_camino_dinamico(self) -> SegmentationConfig:
        if self.motion_threshold < self.velocity_threshold:
            msg = (
                f"motion_threshold ({self.motion_threshold}) es menor que "
                f"velocity_threshold ({self.velocity_threshold}): un mismo frame "
                "contaría a la vez como quietud para STABLE y como movimiento "
                "para el candidato dinámico, y los dos caminos dejarían de ser "
                "excluyentes"
            )
            raise ValueError(msg)
        if self.motion_min_ms >= self.motion_max_ms:
            msg = (
                f"motion_min_ms ({self.motion_min_ms}) no es menor que "
                f"motion_max_ms ({self.motion_max_ms}): todo candidato se "
                "descartaría por largo en el mismo frame en que nace"
            )
            raise ValueError(msg)
        if self.motion_confirm_low_ms < self.stable_ms:
            msg = (
                f"motion_confirm_low_ms ({self.motion_confirm_low_ms}) es menor "
                f"que stable_ms ({self.stable_ms}): el reposo que cierra un trazo "
                "sería más corto que el que la máquina ya llama quietud, y una "
                "pausa que para el camino estático no es una letra partiría una "
                "dinámica en dos"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _coherencia_entre_umbrales(self) -> SegmentationConfig:
        if self.stable_ms > self.buffer_ms:
            msg = (
                f"stable_ms ({self.stable_ms}) excede buffer_ms "
                f"({self.buffer_ms}): nunca se alcanzaría STABLE"
            )
            raise ValueError(msg)
        if self.high_confidence < self.min_confidence:
            msg = (
                f"high_confidence ({self.high_confidence}) es menor que "
                f"min_confidence ({self.min_confidence}): el umbral que emite sin "
                "esperar no puede estar por debajo del piso que decide si se "
                "emite en absoluto"
            )
            raise ValueError(msg)
        if self.reject_cooldown_ms > self.emit_cooldown_ms:
            msg = (
                f"reject_cooldown_ms ({self.reject_cooldown_ms}) excede "
                f"emit_cooldown_ms ({self.emit_cooldown_ms}): un rechazo "
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

    #: Ausencia de mano que cierra la palabra en curso, en **milisegundos**. Un
    #: segundo. Es el único gesto de control del proyecto: bajar la mano entre
    #: palabras es lo que se hace de todos modos, y el clasificador no tiene
    #: clases libres para un gesto dedicado.
    #:
    #: En milisegundos y no en frames por lo mismo que la sección de
    #: segmentación: en frames, "un segundo" eran 1685 ms en la máquina medida.
    space_after_absent_ms: float = Field(default=1000.0, gt=0.0, le=60000.0)


class TelemetryConfig(_Section):
    """Instrumentación de la tasa de cuadros (`src/lsm/telemetry.py`).

    Existe porque todos los umbrales de `segmentation` están en **frames** y sus
    comentarios los traducen a milisegundos suponiendo 30 fps, suposición que no
    se había medido nunca. Ver `docs/adr/0013-la-ventana-mezclada.md`.
    """

    #: Cuadros sobre los que se promedia el fps que se muestra en vivo en el
    #: preview. Es una media móvil a propósito: con la sesión entera, un arranque
    #: lento seguiría tirando del número diez minutos después y el HUD dejaría de
    #: servir para ver el efecto de lo que se acaba de tocar. A 30 fps, 30
    #: cuadros es un segundo de historia.
    fps_window_frames: int = Field(default=30, ge=1, le=3600)

    #: Tasa por debajo de la cual el preview avisa de que el problema es de
    #: rendimiento y no de la seña.
    #:
    #: A tasas muy bajas ninguna ventana temporal razonable contiene frames
    #: suficientes: con `stable_ms = 167`, a 12 fps la estabilidad se decide con
    #: 2 cuadros y a 6 fps con 1, que es no decidir nada. 12 va justo por debajo
    #: del percentil 5 medido en la máquina de referencia (13.8 fps), así que
    #: avisa cuando esa máquina se degrada por debajo de su propio peor caso
    #: normal y no en cada bache.
    min_fps: float = Field(default=12.0, gt=0.0, le=1000.0)

    #: Cuánto dura `lsm-demo --medir-fps` cuando no se le pasa `--medir-segundos`.
    #: Un minuto: lo bastante para que la distribución tenga cola —el percentil 5
    #: de 1800 cuadros son los 90 peores— y lo bastante corto para repetirlo tras
    #: cada ajuste.
    benchmark_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)


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


class SignsConfig(_Section):
    """Dirección texto → señas (`src/lsm/signs.py`, `ARQUITECTURA.md` §4.10).

    No hay clasificador en esta dirección; lo que hay son tiempos. Todos en
    milisegundos a velocidad 1.0: el reproductor los multiplica por la velocidad
    que la persona elija con `+`/`-`.
    """

    #: Cuánto se sostiene en pantalla una letra estática.
    static_hold_ms: float = Field(default=1500.0, gt=0.0, le=60000.0)

    #: Vueltas completas del GIF de una letra dinámica. La duración del paso es
    #: `duracion_ms` del manifest por este número: el movimiento se ve entero
    #: tantas veces como diga. Con dos vueltas y `render_fps` a 12, una letra
    #: dinámica tardaba 15 s en pasar; a 1 vuelta son ~5 s (ver `render_fps`).
    dynamic_loops: int = Field(default=1, ge=1, le=20)

    #: Pausa entre palabras: lo que ocupa un espacio del texto.
    word_gap_ms: float = Field(default=800.0, gt=0.0, le=60000.0)

    #: Cuadros por segundo del GIF al renderizar. Fija `duracion_ms` de cada
    #: dinámica: `round(1000 · frames / render_fps)`. 18, no 12: es la tasa que
    #: `lsm-demo --medir-fps` midió en la Fase 3 (17.8 fps, ADR 0013), así que
    #: el GIF reproduce el trazo a la velocidad real de la grabación en vez de
    #: más lento.
    render_fps: int = Field(default=18, ge=1, le=60)

    #: Lado, en píxeles, del lienzo cuadrado de cada asset.
    canvas_px: int = Field(default=320, ge=64, le=2048)

    #: Aire alrededor de la mano, como fracción del lienzo por cada lado.
    canvas_margin: float = Field(default=0.12, ge=0.0, lt=0.5)

    #: Límites y paso de la velocidad de reproducción.
    speed_min: float = Field(default=0.25, gt=0.0, le=10.0)
    speed_max: float = Field(default=4.0, gt=0.0, le=10.0)
    speed_step: float = Field(default=0.25, gt=0.0, le=10.0)

    #: Espera de `waitKey` en cada vuelta del bucle de la ventana. No es el
    #: `dt` del reproductor: ese se mide con el reloj, este solo decide cada
    #: cuánto se mira el teclado.
    tick_ms: int = Field(default=33, ge=1, le=1000)

    @model_validator(mode="after")
    def _la_velocidad_tiene_rango(self) -> SignsConfig:
        if self.speed_min >= self.speed_max:
            msg = (
                f"signs.speed_min ({self.speed_min}) no es menor que "
                f"signs.speed_max ({self.speed_max}): no hay velocidad válida"
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
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    signs: SignsConfig = Field(default_factory=SignsConfig)

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
        espacio = self.spelling.space_after_absent_ms
        idle = self.segmentation.missing_to_idle_ms
        if espacio <= idle:
            msg = (
                f"spelling.space_after_absent_ms ({espacio}) no supera "
                f"segmentation.missing_to_idle_ms ({idle}): un parpadeo "
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
