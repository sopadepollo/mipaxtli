"""Clasificador de letras estáticas: vecino más cercano por centroides.

`ARQUITECTURA.md` §4.3. Promedia los frames de la ventana con la agregación del
`feature-spec.md` §2 —eso ya lo hace `features.aggregate_static`— y compara el
vector resultante contra un centroide por clase. Nada más.

**Por qué centroides y no k-NN sobre todas las muestras.** El modelo tiene que
viajar a un navegador móvil como JSON y ejecutarse ahí sin dependencias. Un k-NN
de verdad obliga a enviar el dataset entero y a recorrerlo en cada frame; un
centroide por clase son 22 vectores de 42 números, unos 10 KB, y la predicción
son 22 restas. Con un dataset pequeño y clases compactas la diferencia de
precisión es menor que la de tener que depurar por qué el celular va a 4 fps.

## Las tres puertas de rechazo

Un clasificador de 22 clases siempre devuelve una de las 22, aunque la persona se
esté rascando la nariz. `UNKNOWN` es un valor de primera clase (`ARQUITECTURA.md`
§4.4) y aquí se llega a él por tres caminos distintos, en este orden:

1. **σ > `quality.max_dispersion`** — la ventana no era estable: la mano se movía
   o el detector saltó. Se decide *antes* de mirar centroide alguno, porque el
   vector promedio de una ventana temblorosa no representa ninguna configuración.
2. **d₁ > `static_knn.max_distance`** — no se parece a ninguna letra. Sin esta
   puerta, una mano saludando se asigna a la clase que resulte menos lejana.
3. **confianza < `static_knn.min_margin`** — dos centroides igual de cerca. Es el
   caso de las confundibles del §4.8: M y N, S y T. Una duda no es una letra.

## La confianza

`conf = d₂ / (d₁ + d₂)`, con d₁ ≤ d₂ las dos distancias menores. Vale 0.5 con dos
clases empatadas y tiende a 1 cuando la más cercana gana con holgura, así que vive
en `[0.5, 1]` y se puede comparar directamente con `segmentation.min_confidence`.

Se prefirió a un softmax porque no introduce una temperatura que calibrar y porque
se reimplementa en JavaScript en una línea. Se prefirió a `1 − d₁/d_max` porque esa
forma depende de la escala absoluta de las distancias, que cambia con la métrica.

## Pureza

Código puro: sin disco, sin cámara, sin MediaPipe (`CLAUDE.md` §2). Quien lo
entrena y quien escribe el JSON es `lsm.cli.train`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from lsm.classifiers.base import build_export, check_export_compatibility
from lsm.config import Config, Metric
from lsm.features import SequenceFeatures, extract_sequence_features
from lsm.types import NUM_FEATURES, Prediction, Sample, Sequence

#: Nombre con el que este clasificador se identifica en el campo `classifier` del
#: export. Lo lee el `registry` de la Fase 3 para saber qué implementación cargar.
CLASSIFIER_NAME: Final = "static_knn"

__all__ = [
    "CLASSIFIER_NAME",
    "Metric",
    "Ranking",
    "StaticKnnClassifier",
    "Thresholds",
    "apply_thresholds",
    "centroid",
    "decide",
    "distance",
    "rank",
]


# --------------------------------------------------------------------------- #
# Decisión — funciones puras sobre vectores
# --------------------------------------------------------------------------- #
#
# La decisión vive en funciones libres y no solo dentro del método `predict` por
# un motivo concreto: `lsm.evaluation` recorre un barrido de miles de
# combinaciones de umbrales sobre las mismas distancias ya calculadas. Si tuviera
# que pasar por `predict` volvería a extraer features en cada punto de la
# rejilla, y si las reimplementara tendríamos dos reglas de decisión que se
# separarían en cuanto alguien tocara una.


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Todo lo que hace falta para decidir, separado de cómo se entrenó.

    Es lo que viaja en `params` del export y lo que la reimplementación de
    JavaScript necesita copiar.
    """

    metric: Metric
    max_distance: float
    min_margin: float
    max_dispersion: float

    @classmethod
    def from_config(cls, config: Config) -> Thresholds:
        return cls(
            metric=config.static_knn.metric,
            max_distance=config.static_knn.max_distance,
            min_margin=config.static_knn.min_margin,
            max_dispersion=config.quality.max_dispersion,
        )


def distance(
    left: tuple[float, ...], right: tuple[float, ...], metric: Metric
) -> float:
    """Distancia entre dos vectores de features.

    La euclidiana recorre las componentes en orden ascendente y saca la raíz al
    final, igual que el §5.4 exige en el resto de la tubería: la suma en punto
    flotante no es asociativa y la implementación de TypeScript tiene que dar los
    mismos bits.

    En la coseno, un vector de norma cero devuelve la distancia máxima (2.0) en
    vez de dividir por cero. No es un caso hipotético: el paso 7 conserva las
    componentes constantes, pero un centroide de una sola muestra degenerada
    podría anularse.
    """
    if len(left) != len(right):
        msg = f"vectores de distinta longitud: {len(left)} y {len(right)}"
        raise ValueError(msg)

    if metric is Metric.EUCLIDEAN:
        total = 0.0
        for a, b in zip(left, right, strict=True):
            gap = a - b
            total += gap * gap
        return math.sqrt(total)

    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_sq += a * a
        right_sq += b * b
    if left_sq <= 0.0 or right_sq <= 0.0:
        return 2.0
    return 1.0 - dot / (math.sqrt(left_sq) * math.sqrt(right_sq))


def centroid(vectors: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """Media componente a componente. Es todo el "entrenamiento" que hay aquí."""
    if not vectors:
        raise ValueError("no se puede promediar una clase sin muestras")
    count = len(vectors)
    means: list[float] = []
    for component in range(len(vectors[0])):
        total = 0.0
        for vector in vectors:
            total += vector[component]
        means.append(total / count)
    return tuple(means)


@dataclass(frozen=True, slots=True)
class Ranking:
    """Quién ganó y por cuánto, **antes** de aplicar ningún umbral.

    La separación es lo que hace viable el barrido de `lsm-eval`: el orden de las
    distancias no depende de los umbrales, así que se calcula una vez por muestra
    y se reutiliza en los miles de combinaciones de la rejilla. Sin ella, calibrar
    obligaría a recorrer los centroides otra vez en cada punto.
    """

    label: str
    nearest: float
    confidence: float


def rank(distances: Mapping[str, float]) -> Ranking | None:
    """Ordena las distancias y calcula la confianza. `None` si no hay centroides.

    Desempata por orden alfabético de la etiqueta: dos clases exactamente a la
    misma distancia son un empate que la puerta del margen rechazará de todos
    modos, pero el resultado tiene que ser el mismo en dos ejecuciones o el
    reporte de `make eval` deja de ser reproducible.
    """
    if not distances:
        return None

    ordenadas = sorted(distances.items(), key=lambda item: (item[1], item[0]))
    label, nearest = ordenadas[0]

    if len(ordenadas) == 1:
        confidence = 1.0
    else:
        total = nearest + ordenadas[1][1]
        confidence = 1.0 if total <= 0.0 else ordenadas[1][1] / total

    return Ranking(label=label, nearest=nearest, confidence=confidence)


def apply_thresholds(
    ranking: Ranking | None, dispersion: float, thresholds: Thresholds
) -> Prediction:
    """Las tres puertas de rechazo, en orden.

    La de dispersión va primero y no mira el ranking: el vector promedio de una
    ventana temblorosa no representa ninguna configuración, así que la distancia
    a los centroides no significa nada y no hay por qué calcularla.
    """
    if dispersion > thresholds.max_dispersion:
        return Prediction.unknown()
    if ranking is None:
        return Prediction.unknown()
    if ranking.nearest > thresholds.max_distance:
        return Prediction.unknown(confidence=ranking.confidence)
    if ranking.confidence < thresholds.min_margin:
        return Prediction.unknown(confidence=ranking.confidence)
    return Prediction(label=ranking.label, confidence=ranking.confidence)


def decide(
    distances: Mapping[str, float], dispersion: float, thresholds: Thresholds
) -> Prediction:
    """Ranking más umbrales: la decisión completa desde las distancias crudas."""
    return apply_thresholds(rank(distances), dispersion, thresholds)


# --------------------------------------------------------------------------- #
# El clasificador
# --------------------------------------------------------------------------- #


@dataclass
class StaticKnnClassifier:
    """Implementación del `Protocol` de `classifiers/base.py` para estáticas.

    **Es agnóstico a las etiquetas.** No sabe cuáles son dinámicas ni le importa:
    clasifica lo que se le entrene. Restringir la Fase 2 a las estáticas más
    `NONE` es una decisión de alcance de la evaluación y vive en un solo sitio,
    `lsm.evaluation.PHASE2_LABELS`, que es lo que consumen `lsm-train` y
    `lsm-eval`. Meterla aquí dentro duplicaría esa regla en el clasificador y en
    el evaluador, y la Fase 5 tendría que deshacerla en los dos.
    """

    config: Config

    _centroids: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)
    #: Cuántas muestras entraron en cada centroide. Viaja al export porque una
    #: clase entrenada con dos muestras y otra con doscientas no merecen la misma
    #: confianza al leer la matriz de confusión.
    _counts: dict[str, int] = field(default_factory=dict, repr=False)
    #: Muestras que la extracción rechazó al entrenar (escala degenerada). No es
    #: un error: se cuentan y `lsm-train` las reporta.
    rejected: int = 0

    # -- entrenamiento ----------------------------------------------------- #

    def fit(self, samples: list[Sample]) -> None:
        """Un centroide por etiqueta. Reemplaza lo que hubiera entrenado antes."""
        agrupadas: dict[str, list[tuple[float, ...]]] = {}
        rechazadas = 0
        for sample in samples:
            outcome = extract_sequence_features(sample.sequence, self.config)
            if not isinstance(outcome, SequenceFeatures):
                rechazadas += 1
                continue
            agrupadas.setdefault(sample.label, []).append(outcome.static.shape.values)

        self._centroids = {
            label: centroid(tuple(vectores))
            for label, vectores in sorted(agrupadas.items())
        }
        self._counts = {label: len(v) for label, v in sorted(agrupadas.items())}
        self.rejected = rechazadas

    # -- inferencia -------------------------------------------------------- #

    def predict(self, sequence: Sequence) -> Prediction:
        """Clasifica una secuencia `(T, 21, 3)`.

        No lanza nunca: la máquina de estados de la Fase 3 llama a esto treinta
        veces por segundo, y un modelo vacío o una ventana con escala degenerada
        son configuraciones posibles, no errores de programación.
        """
        outcome = extract_sequence_features(sequence, self.config)
        if not isinstance(outcome, SequenceFeatures):
            return Prediction.unknown()
        return self.decide_features(outcome)

    def decide_features(self, features: SequenceFeatures) -> Prediction:
        """La parte de `predict` que va después de extraer.

        Existe aparte para que quien ya tenga las features —la máquina de estados,
        que las necesita igualmente para la velocidad— no las vuelva a calcular.
        """
        thresholds = Thresholds.from_config(self.config)
        distances = {
            label: distance(features.static.shape.values, vector, thresholds.metric)
            for label, vector in self._centroids.items()
        }
        return decide(distances, features.static.dispersion, thresholds)

    # -- export ------------------------------------------------------------ #

    def export(self) -> dict[str, Any]:
        """El JSON del `ARQUITECTURA.md` §4.6, reimplementable en JavaScript.

        `params` lleva los tres umbrales de decisión **y** `smoothing_alpha`. Ese
        último no decide nada, pero sin él la app web no puede reproducir el
        vector de entrada: el suavizado del `feature-spec.md` §4 se aplica antes
        del paso 1, y si el navegador no lo aplica igual el modelo recibe otros
        números (`ARQUITECTURA.md` §4.5).
        """
        return build_export(
            classifier=CLASSIFIER_NAME,
            labels=sorted(self._centroids),
            params={
                "metric": str(self.config.static_knn.metric),
                "max_distance": self.config.static_knn.max_distance,
                "min_margin": self.config.static_knn.min_margin,
                "max_dispersion": self.config.quality.max_dispersion,
                "smoothing_alpha": self.config.smoothing.alpha,
            },
            data={
                "centroids": {
                    label: [float(value) for value in vector]
                    for label, vector in sorted(self._centroids.items())
                },
                "sample_counts": dict(sorted(self._counts.items())),
            },
        )

    @classmethod
    def from_export(cls, payload: Mapping[str, Any]) -> StaticKnnClassifier:
        """Reconstruye el clasificador desde su JSON.

        Es la prueba ejecutable de que el export está completo: si al recargar
        hiciera falta algo que no viaja en el archivo, la app web tampoco lo
        tendría y el fallo aparecería en la Fase 7 en vez de aquí.

        Verifica la compatibilidad **antes** de leer nada. Un modelo entrenado con
        otra normalización no falla ruidosamente: acierta menos, y eso es
        exactamente lo que no se detecta en una demo (`classifiers/base.py`).
        """
        check_export_compatibility(payload)
        if payload["classifier"] != CLASSIFIER_NAME:
            msg = (
                f"el modelo dice ser {payload['classifier']!r}, no {CLASSIFIER_NAME!r}"
            )
            raise ValueError(msg)

        params = payload["params"]
        config = Config.model_validate(
            {
                "static_knn": {
                    "metric": params["metric"],
                    "max_distance": params["max_distance"],
                    "min_margin": params["min_margin"],
                },
                "quality": {"max_dispersion": params["max_dispersion"]},
                "smoothing": {"alpha": params["smoothing_alpha"]},
            }
        )

        centroids: dict[str, tuple[float, ...]] = {}
        for label, vector in payload["data"]["centroids"].items():
            if len(vector) != NUM_FEATURES:
                msg = (
                    f"el centroide de {label} trae {len(vector)} componentes, "
                    f"no {NUM_FEATURES}"
                )
                raise ValueError(msg)
            centroids[label] = tuple(float(value) for value in vector)

        classifier = cls(config=config)
        classifier._centroids = dict(sorted(centroids.items()))
        classifier._counts = dict(payload["data"].get("sample_counts", {}))
        return classifier
