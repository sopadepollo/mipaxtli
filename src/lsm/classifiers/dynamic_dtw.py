"""Clasificador de letras dinámicas: DTW contra plantillas de referencia.

`ARQUITECTURA.md` §4.3 y `docs/feature-spec.md` §3. La ventana que llega es el
trazo crudo que la segmentación acumuló en DYNAMIC_CANDIDATE (§6.7); aquí se
extrae con la tubería de siempre, se remuestrea a `T_ref = 24` filas de
`g_t ∈ ℝ⁴⁵` (§3.2-§3.3) y se alinea contra cada plantilla con DTW (§3.4).

**Por qué DTW y no una red.** Con tres personas grabadas una red recurrente
sobreajusta, y además hay que ejecutarla en un navegador móvil. DTW no entrena
nada, tolera que la misma letra se trace más rápido o más lento, se depura
mirando una matriz de costos y se reimplementa en TypeScript en unas sesenta
líneas. Ver `docs/adr/0016-clasificador-dinamico-y-barrido.md`.

## Las plantillas

Una por **(letra, persona)**: el medoide DTW de las muestras de esa persona para
esa letra, o sea la muestra cuya suma de distancias al resto de su grupo es
mínima. Es una muestra real, no un promedio: promediar trazos con ritmos
distintos da un trazo que nadie ejecutó, que es justo lo que DTW existe para no
tener que hacer. Por persona y no por letra porque cada quien traza a su manera,
y quedarse con una sola plantilla por letra obligaría a elegir de quién.

Las letras de `vocabulary.DIRECTION_PENDING_LABELS` —`LL` y `RR`— **no tienen
plantilla**: su dirección canónica está pendiente de decisión humana, y con los
dos sentidos mezclados el medoide sería el de uno de los dos al azar. Se cuentan
en `blocked` y el export las rechaza al cargar.

## La decisión

`d_label = min` sobre las plantillas de la letra, y luego la misma regla que
`static_knn`: `conf = d₂ / (d₁ + d₂)` entre las dos letras más cercanas
(`static_knn.rank`, reutilizada y no reescrita), y **UNKNOWN si
`d₁ > dtw.max_distance`**. DTW da una distancia, no una probabilidad, y siempre
tiene una plantilla más cercana: sin esa puerta, cualquier movimiento que se
cuele al camino dinámico se forzaría a la letra menos lejana.

## Pureza

Código puro: sin disco, sin cámara, sin MediaPipe (`CLAUDE.md` §2). Quien lo
entrena y quien escribe el JSON es `lsm.cli.train`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass, field
from typing import Any, Final

from lsm.classifiers.base import build_export, check_export_compatibility
from lsm.classifiers.static_knn import Ranking, rank
from lsm.config import Config
from lsm.features import (
    RESAMPLE_LENGTH,
    DynamicFeatures,
    SequenceFeatures,
    extract_sequence_features,
)
from lsm.preprocessing import config_from_record, preprocessing_record
from lsm.types import NUM_FEATURES, Prediction, Sample, Sequence
from lsm.vocabulary import DIRECTION_PENDING_LABELS

#: Nombre con el que este clasificador se identifica en el campo `classifier` del
#: export. Es lo que `classifiers.registry` lee para saber qué cargar.
CLASSIFIER_NAME: Final = "dynamic_dtw"

#: Componentes de `g_t` (§3.3): 42 de forma, 2 de trayectoria y δ, ponderados.
DYNAMIC_ROW_WIDTH: Final = NUM_FEATURES + 3

#: Una fila de `g_t` y una secuencia remuestreada de ellas.
Row = tuple[float, ...]
Rows = tuple[Row, ...]

__all__ = [
    "CLASSIFIER_NAME",
    "DYNAMIC_ROW_WIDTH",
    "DynamicDtwClassifier",
    "Template",
    "Thresholds",
    "apply_thresholds",
    "dtw_distance",
    "dynamic_rows",
    "label_distances",
    "local_distance",
    "medoid",
]


# --------------------------------------------------------------------------- #
# §3.4 — La distancia
# --------------------------------------------------------------------------- #


def local_distance(left: Row, right: Row) -> float:
    """Distancia euclidiana entre dos filas de `g_t`.

    La suma recorre las componentes en orden ascendente y la raíz va al final,
    igual que el resto de la tubería (§5, regla 4). No se usa `sum()` a
    propósito: desde Python 3.12 suma los flotantes con compensación de Neumaier,
    que no es lo que hará TypeScript con un bucle.
    """
    total = 0.0
    for a, b in zip(left, right, strict=True):
        gap = a - b
        total += gap * gap
    return math.sqrt(total)


def dtw_distance(left: Rows, right: Rows, band_radius: int) -> float:
    """DTW con banda de Sakoe-Chiba, normalizado por la longitud del camino.

    `feature-spec.md` §3.4, escrito aquí con el detalle que la paridad exige:

    - Celdas admitidas: `|i − j| ≤ band_radius`.
    - Recurrencia: `D[i][j] = c(i, j) + min(D[i−1][j−1], D[i−1][j], D[i][j−1])`,
      con `D[0][0] = c(0, 0)` y `c` la `local_distance`.
    - **Desempate**, que decide qué camino se cuenta: diagonal, luego `(i−1, j)`,
      luego `(i, j−1)`. Solo gana un predecesor si su costo es **estrictamente**
      menor que el del anterior en ese orden.
    - `L[i][j]` es el número de celdas del camino elegido; el resultado es
      `D[n−1][m−1] / L[n−1][m−1]`.

    Normalizar por la longitud del camino hace comparables los costos de dos
    alineaciones que usaron distinto número de pasos: sin eso, el camino
    diagonal —el más corto— saldría favorecido solo por sumar menos términos.

    Las dos secuencias tienen que medir lo mismo: después del §3.2 siempre son
    `T_ref` filas, y admitir longitudes distintas obligaría a definir una banda
    inclinada que ninguna de las dos implementaciones necesita.
    """
    if not left or not right:
        raise ValueError("no se puede alinear una secuencia vacía")
    if len(left) != len(right):
        msg = (
            f"secuencias de distinta longitud ({len(left)} y {len(right)}): el "
            "DTW del §3.4 alinea secuencias ya remuestreadas a T_ref"
        )
        raise ValueError(msg)
    if band_radius < 0:
        raise ValueError(f"band_radius tiene que ser ≥ 0, no {band_radius}")

    length = len(left)
    cost = [[math.inf] * length for _ in range(length)]
    steps = [[0] * length for _ in range(length)]

    for i in range(length):
        for j in range(max(0, i - band_radius), min(length, i + band_radius + 1)):
            local = local_distance(left[i], right[j])
            if i == 0 and j == 0:
                cost[0][0] = local
                steps[0][0] = 1
                continue

            best = math.inf
            best_steps = 0
            # Diagonal, vertical, horizontal: el orden ES el desempate.
            if i > 0 and j > 0 and cost[i - 1][j - 1] < best:
                best = cost[i - 1][j - 1]
                best_steps = steps[i - 1][j - 1]
            if i > 0 and cost[i - 1][j] < best:
                best = cost[i - 1][j]
                best_steps = steps[i - 1][j]
            if j > 0 and cost[i][j - 1] < best:
                best = cost[i][j - 1]
                best_steps = steps[i][j - 1]

            cost[i][j] = best + local
            steps[i][j] = best_steps + 1

    return cost[length - 1][length - 1] / steps[length - 1][length - 1]


# --------------------------------------------------------------------------- #
# Plantillas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Template:
    """Una plantilla de referencia: la muestra medoide de una (letra, persona)."""

    label: str
    signer_id: str
    rows: Rows


def medoid(group: AbcSequence[Rows], band_radius: int) -> int:
    """Índice del medoide DTW de un grupo: la suma de distancias mínima.

    Empates por el índice más bajo, así que el resultado depende del orden en
    que llegan las muestras; quien llama lo fija (`fit` ordena por sesión y
    marca de tiempo). Con una sola muestra, el medoide es ella misma.

    El DTW con esta recurrencia **no** es exactamente simétrico —el desempate
    distingue vertical de horizontal—, así que se calculan los dos sentidos y no
    se reutiliza `d(a, b)` como `d(b, a)`. Cuesta el doble y quita una
    suposición falsa del camino.
    """
    if not group:
        raise ValueError("no hay medoide de un grupo vacío")
    best_index = 0
    best_total = math.inf
    for index, candidate in enumerate(group):
        total = 0.0
        for other_index, other in enumerate(group):
            if other_index != index:
                total += dtw_distance(candidate, other, band_radius)
        if total < best_total:
            best_total = total
            best_index = index
    return best_index


def label_distances(
    rows: Rows, templates: AbcSequence[Template], band_radius: int
) -> dict[str, float]:
    """La distancia de una secuencia a cada letra: el mínimo sobre sus plantillas."""
    distances: dict[str, float] = {}
    for template in templates:
        d = dtw_distance(rows, template.rows, band_radius)
        if d < distances.get(template.label, math.inf):
            distances[template.label] = d
    return distances


# --------------------------------------------------------------------------- #
# Decisión
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Lo que decide, separado de cómo se entrenó. Viaja en `params`."""

    band_radius: int
    max_distance: float

    @classmethod
    def from_config(cls, config: Config) -> Thresholds:
        return cls(
            band_radius=config.dtw.band_radius,
            max_distance=config.dtw.max_distance,
        )


def apply_thresholds(ranking: Ranking | None, max_distance: float) -> Prediction:
    """La puerta de distancia: lo que no se parece a ninguna plantilla, UNKNOWN.

    No hay puerta de margen propia: la confianza sale de `static_knn.rank`, en
    la misma escala `[0.5, 1]`, y el piso que decide si se escribe es
    `segmentation.min_confidence`, igual que para las estáticas.
    """
    if ranking is None:
        return Prediction.unknown()
    if ranking.nearest > max_distance:
        return Prediction.unknown(confidence=ranking.confidence)
    return Prediction(label=ranking.label, confidence=ranking.confidence)


# --------------------------------------------------------------------------- #
# El clasificador
# --------------------------------------------------------------------------- #


@dataclass
class DynamicDtwClassifier:
    """Implementación del `Protocol` de `classifiers/base.py` para dinámicas.

    Agnóstico a las etiquetas, como `static_knn`: clasifica lo que se le entrene.
    La única regla de vocabulario que aplica por su cuenta es el bloqueo de
    `DIRECTION_PENDING_LABELS`, porque es una prohibición y no un alcance: no
    hay llamada correcta que deba poder saltársela.
    """

    config: Config

    _templates: list[Template] = field(default_factory=list, repr=False)
    _counts: dict[str, int] = field(default_factory=dict, repr=False)
    #: Muestras que la extracción rechazó o que no daban para canal dinámico.
    rejected: int = 0
    #: Muestras descartadas por etiqueta bloqueada, por etiqueta.
    blocked: dict[str, int] = field(default_factory=dict)

    # -- entrenamiento ----------------------------------------------------- #

    def fit(self, samples: list[Sample]) -> None:
        """Una plantilla por (letra, persona): su medoide DTW.

        Las muestras se ordenan por sesión y marca de tiempo antes de agrupar,
        para que el desempate del medoide no dependa del orden en que las lea
        el disco.
        """
        band = self.config.dtw.band_radius
        grupos: dict[tuple[str, str], list[Rows]] = {}
        rechazadas = 0
        bloqueadas: dict[str, int] = {}

        ordenadas = sorted(
            samples, key=lambda s: (s.label, s.signer_id, s.session_id, s.timestamp)
        )
        for sample in ordenadas:
            if sample.label in DIRECTION_PENDING_LABELS:
                bloqueadas[sample.label] = bloqueadas.get(sample.label, 0) + 1
                continue
            rows = dynamic_rows(sample.sequence, self.config)
            if rows is None:
                rechazadas += 1
                continue
            grupos.setdefault((sample.label, sample.signer_id), []).append(rows)

        self._templates = [
            Template(label=label, signer_id=signer, rows=grupo[medoid(grupo, band)])
            for (label, signer), grupo in sorted(grupos.items())
        ]
        counts: dict[str, int] = {}
        for (label, _), grupo in grupos.items():
            counts[label] = counts.get(label, 0) + len(grupo)
        self._counts = dict(sorted(counts.items()))
        self.rejected = rechazadas
        self.blocked = dict(sorted(bloqueadas.items()))

    @property
    def templates(self) -> tuple[Template, ...]:
        return tuple(self._templates)

    # -- inferencia -------------------------------------------------------- #

    def predict(self, sequence: Sequence) -> Prediction:
        """Clasifica un trazo `(T, 21, 3)`. No lanza nunca.

        Un trazo demasiado corto para el §3.2 o con escala degenerada es UNKNOWN,
        no un error: la máquina de estados puede entregar cualquiera de los dos.
        """
        rows = dynamic_rows(sequence, self.config)
        if rows is None:
            return Prediction.unknown()
        return self.decide_rows(rows)

    def decide_rows(self, rows: Rows) -> Prediction:
        """La parte de `predict` que va después de extraer y remuestrear."""
        thresholds = Thresholds.from_config(self.config)
        distances = label_distances(rows, self._templates, thresholds.band_radius)
        return apply_thresholds(rank(distances), thresholds.max_distance)

    # -- export ------------------------------------------------------------ #

    def export(self) -> dict[str, Any]:
        """El JSON del `ARQUITECTURA.md` §4.6, reimplementable en TypeScript.

        `params` lleva lo que decide —`band_radius`, `max_distance`— y lo que hace
        falta para reconstruir la entrada: `trajectory_weight` y `depth_weight`
        (§3.3),
        `min_source_frames` y `resample_length` (§3.2) y `preprocessing` (la
        plausibilidad, el relleno y el One Euro, §0.3, §0.4 y §4). Sin cualquiera
        de ellos, el navegador compararía otras filas contra las mismas
        plantillas.
        """
        return build_export(
            classifier=CLASSIFIER_NAME,
            labels=sorted({t.label for t in self._templates}),
            params={
                "band_radius": self.config.dtw.band_radius,
                "max_distance": self.config.dtw.max_distance,
                "trajectory_weight": self.config.features.trajectory_weight,
                "depth_weight": self.config.features.depth_weight,
                "min_source_frames": self.config.dtw.min_source_frames,
                "resample_length": RESAMPLE_LENGTH,
                "preprocessing": preprocessing_record(self.config),
            },
            data={
                "templates": [
                    {
                        "label": t.label,
                        "signer_id": t.signer_id,
                        "rows": [[float(v) for v in row] for row in t.rows],
                    }
                    for t in self._templates
                ],
                "sample_counts": dict(self._counts),
                "blocked_labels": sorted(DIRECTION_PENDING_LABELS),
            },
        )

    @classmethod
    def from_export(cls, payload: Mapping[str, Any]) -> DynamicDtwClassifier:
        """Reconstruye el clasificador desde su JSON, verificando antes de leer.

        Además de las versiones, rechaza un modelo que traiga plantillas de una
        letra bloqueada: uno entrenado antes del bloqueo, o a mano, llevaría una
        dirección que nadie decidió.
        """
        check_export_compatibility(payload)
        if payload["classifier"] != CLASSIFIER_NAME:
            msg = (
                f"el modelo dice ser {payload['classifier']!r}, no {CLASSIFIER_NAME!r}"
            )
            raise ValueError(msg)

        params = payload["params"]
        if params["resample_length"] != RESAMPLE_LENGTH:
            msg = (
                f"el modelo remuestrea a {params['resample_length']} filas y este "
                f"runtime a {RESAMPLE_LENGTH}"
            )
            raise ValueError(msg)

        config = Config.model_validate(
            {
                "dtw": {
                    "band_radius": params["band_radius"],
                    "max_distance": params["max_distance"],
                    "min_source_frames": params["min_source_frames"],
                },
                "features": {
                    "trajectory_weight": params["trajectory_weight"],
                    "depth_weight": params["depth_weight"],
                },
                **config_from_record(params["preprocessing"]),
            }
        )

        templates: list[Template] = []
        for entry in payload["data"]["templates"]:
            label = entry["label"]
            if label in DIRECTION_PENDING_LABELS:
                msg = (
                    f"el modelo trae plantilla de {label}, cuya dirección canónica "
                    "está pendiente de decisión humana (docs/glosario-lsm.md); "
                    "reentrenar"
                )
                raise ValueError(msg)
            rows = tuple(tuple(float(v) for v in row) for row in entry["rows"])
            if len(rows) != RESAMPLE_LENGTH or any(
                len(row) != DYNAMIC_ROW_WIDTH for row in rows
            ):
                msg = (
                    f"la plantilla de {label} no es ({RESAMPLE_LENGTH}, "
                    f"{DYNAMIC_ROW_WIDTH})"
                )
                raise ValueError(msg)
            templates.append(
                Template(label=label, signer_id=entry["signer_id"], rows=rows)
            )

        classifier = cls(config=config)
        classifier._templates = templates
        classifier._counts = dict(payload["data"].get("sample_counts", {}))
        return classifier


def dynamic_rows(sequence: Sequence, config: Config) -> Rows | None:
    """Las 24 filas de `g_t` de una secuencia, o `None` si no dan para el §3.2.

    `None` cubre los dos motivos —escala degenerada y trazo demasiado corto— y es
    interno a este módulo y a la evaluación: hacia fuera, los dos son UNKNOWN.
    """
    outcome = extract_sequence_features(sequence, config)
    if not isinstance(outcome, SequenceFeatures):
        return None
    if not isinstance(outcome.dynamic, DynamicFeatures):
        return None
    return outcome.dynamic.rows
