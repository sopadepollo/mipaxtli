"""Protocolo de evaluación de la Fase 2: splits, métricas, barrido y contraste.

`src/lsm/cli/evaluate.py` es el envoltorio: lee el dataset del disco, lee la
columna `confundible_con` del glosario y escribe el reporte. Todo lo que **decide
qué significa un número** vive aquí, y por eso se puede ejercitar entero en CI:
el criterio de aceptación de la fase —un reporte reproducible con su barrido y su
contraste de hipótesis— no debería depender de que haya grabaciones en `data/raw`.

Es el mismo motivo por el que `capture.py` existe separado de `cli/capture.py`
(ver `docs/adr/0006-...`), y la misma decisión: la parte que se equivoca en
silencio es la que hay que poder testear.

## Las tres cosas que este módulo no negocia

1. **El alcance es 21 estáticas + `NONE`.** Las ocho dinámicas quedan fuera hasta
   la Fase 5. `static_knn` promedia la secuencia; el promedio de una Z no es una
   configuración de mano, es una mancha, y aterrizará sobre alguna estática
   ensuciando su fila de la matriz. Una matriz de confusión ilegible no se mira, y
   una matriz que no se mira no calibra nada.

2. **El split es por persona, nunca aleatorio** (`CLAUDE.md` §6). Un split
   aleatorio de frames pone frames de la misma grabación en train y en test: el
   modelo reconoce la grabación, no la seña, y el número sale precioso.

3. **Nada de aleatoriedad.** No hay semillas porque no hay sorteos: los folds
   salen de ordenar los identificadores, los empates se rompen alfabéticamente y
   la rejilla se recorre en orden. `make eval` da los mismos bytes dos veces o el
   reporte no sirve para comparar dos calibraciones.

Código puro: sin disco, sin cámara, sin MediaPipe.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from lsm.capture import trajectory_arc_length
from lsm.classifiers.static_knn import (
    Ranking,
    Thresholds,
    apply_thresholds,
    centroid,
    distance,
    rank,
)
from lsm.config import Config, Metric
from lsm.features import SequenceFeatures, extract_sequence_features
from lsm.types import NEGATIVE_LABEL, UNKNOWN_LABEL, Sample, SampleKind
from lsm.vocabulary import DYNAMIC_LABELS, STATIC_LABELS

#: El vocabulario de la Fase 2: las 21 letras sin movimiento más la clase
#: negativa, en orden alfabético con `NONE` al final. Es **la única** definición
#: del alcance; `lsm-train` y `lsm-eval` la consumen para no poder discrepar.
PHASE2_LABELS: Final[tuple[str, ...]] = (
    *sorted(label.value for label in STATIC_LABELS),
    NEGATIVE_LABEL,
)

#: Las ocho que esperan a la Fase 5: J, K, LL, Ñ, Q, RR, X y Z.
EXCLUDED_LABELS: Final[frozenset[str]] = frozenset(
    label.value for label in DYNAMIC_LABELS
)

#: Un par de etiquetas, siempre ordenado. La confusión es simétrica —que A se
#: prediga como E y que E se prediga como A son el mismo fenómeno— y el glosario
#: la anota en las dos filas, así que la clave tiene que serlo también.
Pair = tuple[str, str]


def as_pair(left: str, right: str) -> Pair:
    return (left, right) if left <= right else (right, left)


class Protocol(StrEnum):
    """Cómo se parte el dataset. Nunca aleatoriamente."""

    #: El protocolo del proyecto: un fold por persona (`ARQUITECTURA.md` §4.7).
    SIGNER = "leave-one-signer-out"
    #: Degradación consciente para cuando solo ha grabado una persona. Mide si el
    #: modelo generaliza entre **sesiones**, que es mucho menos de lo que parece:
    #: la misma mano, el mismo estilo, casi la misma iluminación.
    SESSION = "leave-one-session-out"


class InsufficientFoldsError(RuntimeError):
    """El dataset no da para el protocolo pedido."""


# --------------------------------------------------------------------------- #
# Observaciones: el dataset ya reducido a lo que la evaluación necesita
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Observation:
    """Una muestra con sus features ya extraídas.

    Existe para que el barrido no vuelva a recorrer la tubería de features en cada
    punto de la rejilla: extraer es lo caro, y de los miles de puntos solo unos
    pocos cambian algo que afecte a la extracción.

    Los tres números del final no los usa el clasificador. Son el insumo de la
    sección de diagnóstico del reporte, la que responde "¿dónde habría que poner
    `velocity_threshold`?" mirando la distribución real en vez de razonando.
    """

    label: str
    signer_id: str
    session_id: str
    shape: tuple[float, ...]
    dispersion: float
    arc_length: float
    mean_velocity: float
    max_velocity: float


@dataclass(frozen=True, slots=True)
class Dataset:
    """Las observaciones utilizables, más el rastro de lo que se quedó fuera.

    Lo descartado se cuenta y se reporta. Un dataset que perdió el 30% de sus
    muestras por escala degenerada da un accuracy estupendo sobre el 70% que
    quedó, y sin este registro nadie se entera.
    """

    observations: tuple[Observation, ...]
    #: Etiquetas fuera del alcance de la Fase 2 y cuántas muestras se descartaron.
    excluded: dict[str, int]
    #: Muestras cuya extracción de features falló, por motivo.
    rejected: dict[str, int]
    #: Muestras de `NONE` grabadas en modo dinámico. Se cuentan aparte de
    #: `excluded` porque no se descartan por su etiqueta —`NONE` sí está en el
    #: alcance— sino por cómo se grabaron. Ver
    #: `docs/adr/0010-la-clase-negativa-y-el-modo-de-grabacion.md`.
    dynamic_negatives: int = 0

    @property
    def signers(self) -> tuple[str, ...]:
        return tuple(sorted({o.signer_id for o in self.observations}))

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted({o.session_id for o in self.observations}))

    @property
    def labels(self) -> tuple[str, ...]:
        presentes = {o.label for o in self.observations}
        return tuple(label for label in PHASE2_LABELS if label in presentes)


def observe(
    samples: Iterable[Sample], config: Config, labels: tuple[str, ...] = PHASE2_LABELS
) -> Dataset:
    """Extrae las features una vez y filtra al alcance de la fase.

    Una etiqueta dinámica se descarta y se cuenta. Una etiqueta que no es ni
    estática, ni dinámica, ni `NONE` **levanta**: es un dataset corrupto o un
    glosario que cambió bajo los pies del código, y las dos cosas hay que verlas
    ahora y no como una fila rara de la matriz dentro de tres semanas.

    **`NONE` grabada en modo dinámico también se descarta**, y por su `kind`, no
    por su etiqueta. La clase negativa se graba de las dos formas a propósito
    —mano en reposo, y también transiciones y saludos— pero `static_knn` le da un
    único centroide y lo construye promediando los frames de cada muestra: el
    promedio de un saludo no es ninguna configuración de mano. Medido sobre las 83
    muestras de `NONE` de s01-s03: las 32 dinámicas tienen σ mediana 0.1832 contra
    0.0212 de las estáticas, y 28 de ellas superan `quality.max_dispersion`, así
    que además entran rechazadas de oficio. Envenenaban el centroide al entrenar y
    contaban como fallo al evaluar.

    No se pierden: el dato sigue en `data/raw` con su `kind`, y es el material del
    clasificador dinámico de la Fase 5.
    """
    admitidas = set(labels)
    observations: list[Observation] = []
    excluded: dict[str, int] = {}
    rejected: dict[str, int] = {}
    dynamic_negatives = 0

    for sample in samples:
        if sample.label in EXCLUDED_LABELS:
            excluded[sample.label] = excluded.get(sample.label, 0) + 1
            continue
        if sample.label == NEGATIVE_LABEL and sample.kind is SampleKind.DYNAMIC:
            dynamic_negatives += 1
            continue
        if sample.label not in admitidas:
            msg = (
                f"etiqueta desconocida en el dataset: {sample.label!r}. No es una "
                "letra del glosario ni la clase negativa; revisa data/raw o "
                "lsm.vocabulary antes de seguir."
            )
            raise ValueError(msg)

        outcome = extract_sequence_features(sample.sequence, config)
        if not isinstance(outcome, SequenceFeatures):
            motivo = str(outcome.reason)
            rejected[motivo] = rejected.get(motivo, 0) + 1
            continue

        velocities = outcome.velocities
        observations.append(
            Observation(
                label=sample.label,
                signer_id=sample.signer_id,
                session_id=sample.session_id,
                shape=outcome.static.shape.values,
                dispersion=outcome.static.dispersion,
                arc_length=trajectory_arc_length(outcome.trajectory),
                mean_velocity=(
                    sum(velocities) / len(velocities) if velocities else 0.0
                ),
                max_velocity=max(velocities) if velocities else 0.0,
            )
        )

    return Dataset(
        observations=tuple(observations),
        excluded=excluded,
        rejected=rejected,
        dynamic_negatives=dynamic_negatives,
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Fold:
    """Un pliegue: qué se dejó fuera y qué índices caen de cada lado."""

    held_out: str
    train: tuple[int, ...]
    test: tuple[int, ...]


def build_folds(dataset: Dataset, protocol: Protocol) -> tuple[Fold, ...]:
    """Un fold por persona (o por sesión), en orden alfabético del identificador.

    Se niega con un solo grupo en vez de degradar sola al protocolo por sesión.
    Las dos métricas no son comparables —una mide si el modelo generaliza a otra
    persona, la otra si generaliza a otro día de la misma persona— y la diferencia
    se pierde en cuanto el número sale del reporte. Que la degradación la pida
    quien evalúa, y que quede escrita en la cabecera.
    """
    if not dataset.observations:
        raise InsufficientFoldsError("el dataset no tiene ninguna observación")

    def clave(observation: Observation) -> str:
        return (
            observation.signer_id
            if protocol is Protocol.SIGNER
            else observation.session_id
        )

    grupos = sorted({clave(o) for o in dataset.observations})

    if len(grupos) < 2:
        que = "firmante" if protocol is Protocol.SIGNER else "sesión"
        msg = (
            f"{protocol.value} necesita al menos 2 grupos y el dataset tiene "
            f"{len(grupos)}: hay un solo {que} ({grupos[0]!r}). Entrenar y probar "
            "con la misma persona no mide generalización, mide memoria. Con una "
            "sola persona grabada, `--protocolo leave-one-session-out` es la "
            "degradación consciente; el reporte dirá cuál se usó."
        )
        raise InsufficientFoldsError(msg)

    folds: list[Fold] = []
    for grupo in grupos:
        train = tuple(
            índice for índice, o in enumerate(dataset.observations) if clave(o) != grupo
        )
        test = tuple(
            índice for índice, o in enumerate(dataset.observations) if clave(o) == grupo
        )
        folds.append(Fold(held_out=grupo, train=train, test=test))
    return tuple(folds)


# --------------------------------------------------------------------------- #
# Distancias precalculadas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ScoredSample:
    """Una muestra de test con su ranking ya resuelto.

    El orden de las distancias **no depende de los umbrales**, así que se calcula
    una vez y el barrido entero se convierte en comparar tres números por muestra
    en vez de recorrer 22 centroides de 42 componentes. Es lo que hace que la
    rejilla de calibración quepa en `make eval`.
    """

    truth: str
    dispersion: float
    ranking: Ranking | None


@dataclass(frozen=True, slots=True)
class FoldDistances:
    held_out: str
    #: Etiquetas que el fold pudo aprender. Si una clase solo la grabó la persona
    #: que este fold deja fuera, aquí no está y sus muestras no tienen forma de
    #: acertar. Se reporta.
    trained_labels: tuple[str, ...]
    samples: tuple[ScoredSample, ...]


def precompute(
    dataset: Dataset, folds: tuple[Fold, ...], metric: Metric
) -> tuple[FoldDistances, ...]:
    """Entrena un centroide por clase en cada fold y rankea su conjunto de test."""
    resultado: list[FoldDistances] = []
    for fold in folds:
        agrupadas: dict[str, list[tuple[float, ...]]] = {}
        for índice in fold.train:
            observation = dataset.observations[índice]
            agrupadas.setdefault(observation.label, []).append(observation.shape)
        centroides = {
            label: centroid(tuple(vectores))
            for label, vectores in sorted(agrupadas.items())
        }

        muestras = tuple(
            ScoredSample(
                truth=dataset.observations[índice].label,
                dispersion=dataset.observations[índice].dispersion,
                ranking=rank(
                    {
                        label: distance(
                            dataset.observations[índice].shape, vector, metric
                        )
                        for label, vector in centroides.items()
                    }
                ),
            )
            for índice in fold.test
        )
        resultado.append(
            FoldDistances(
                held_out=fold.held_out,
                trained_labels=tuple(sorted(centroides)),
                samples=muestras,
            )
        )
    return tuple(resultado)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LabelScore:
    """Cómo le fue a una letra."""

    total: int
    correct: int
    unknown: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class Report:
    """El resultado de evaluar una configuración sobre unos folds."""

    #: `(etiqueta verdadera, etiqueta predicha) -> cuántas veces`. La predicha
    #: puede ser `UNKNOWN`: un rechazo es un resultado, no una ausencia de dato.
    confusion: dict[Pair, int]
    per_label: dict[str, LabelScore]
    total: int
    correct: int
    unknown: int

    @property
    def accuracy(self) -> float:
        """Aciertos sobre el total. Un `UNKNOWN` cuenta como fallo.

        Es lo severo y es lo correcto para la Fase 2: el criterio de aceptación es
        ≥90% reconociendo letras, y una configuración que se calla siempre no
        reconoce ninguna. La tasa de rechazo se reporta aparte para poder leer las
        dos cosas juntas.
        """
        return self.correct / self.total if self.total else 0.0

    @property
    def macro_accuracy(self) -> float:
        """Media de los accuracy por letra, sin ponderar por número de muestras.

        Es la que hay que mirar si el dataset quedó desbalanceado: con 500
        muestras de `NONE` y 20 de cada letra, el accuracy global se puede
        sostener acertando solo la clase negativa.
        """
        if not self.per_label:
            return 0.0
        return sum(s.accuracy for s in self.per_label.values()) / len(self.per_label)

    @property
    def unknown_rate(self) -> float:
        return self.unknown / self.total if self.total else 0.0


def score(folds: tuple[FoldDistances, ...], thresholds: Thresholds) -> Report:
    """Aplica los umbrales a los rankings ya calculados y acumula la matriz."""
    confusion: dict[Pair, int] = {}
    per_label: dict[str, list[int]] = {}
    total = 0
    correct = 0
    unknown = 0

    for fold in folds:
        for muestra in fold.samples:
            prediction = apply_thresholds(
                muestra.ranking, muestra.dispersion, thresholds
            )
            clave = (muestra.truth, prediction.label)
            confusion[clave] = confusion.get(clave, 0) + 1

            acierto = prediction.label == muestra.truth
            rechazo = prediction.is_unknown
            contador = per_label.setdefault(muestra.truth, [0, 0, 0])
            contador[0] += 1
            contador[1] += int(acierto)
            contador[2] += int(rechazo)

            total += 1
            correct += int(acierto)
            unknown += int(rechazo)

    return Report(
        confusion=dict(sorted(confusion.items())),
        per_label={
            label: LabelScore(total=c[0], correct=c[1], unknown=c[2])
            for label, c in sorted(per_label.items())
        },
        total=total,
        correct=correct,
        unknown=unknown,
    )


def most_confused(
    confusion: Mapping[Pair, int], limit: int = 10
) -> tuple[tuple[Pair, int], ...]:
    """Los pares de letras que más se cruzan, sumando las dos direcciones.

    `UNKNOWN` queda fuera: un rechazo no es una confusión **entre dos letras**, y
    como suele ser la celda más grande de la fila, mezclarlo taparía justo lo que
    esta lista existe para enseñar. La tasa de rechazo se reporta en su sección.
    """
    acumulado: dict[Pair, int] = {}
    for (verdadera, predicha), cuenta in confusion.items():
        if verdadera == predicha or UNKNOWN_LABEL in (verdadera, predicha):
            continue
        par = as_pair(verdadera, predicha)
        acumulado[par] = acumulado.get(par, 0) + cuenta

    ordenado = sorted(
        ((par, cuenta) for par, cuenta in acumulado.items() if cuenta > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(ordenado[:limit])


# --------------------------------------------------------------------------- #
# Contraste de hipótesis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Contrast:
    """Las predicciones del glosario contra la matriz de confusión real.

    La columna `confundible_con` se llenó leyendo las descripciones de *Manos con
    voz*, **sin datos y antes de grabar nada**. Eso la convierte en una hipótesis
    falsable, y compararla con lo que pasó es lo que separa una evaluación de un
    vistazo a una tabla.

    Las cuatro cubetas, y por qué son cuatro y no dos: trece de los veintiséis
    pares del glosario tocan una letra dinámica, que la Fase 2 no evalúa. Meterlos
    entre los refutados diría que el glosario se equivocó cuando lo que ocurre es
    que todavía no se ha medido.
    """

    #: Predicho y ocurrió. La hipótesis se sostiene.
    confirmed: tuple[Pair, ...]
    #: Predicho, evaluable y **no** ocurrió. La intuición sobre la configuración
    #: manual no se tradujo en una confusión del clasificador. Puede ser que las
    #: features separen mejor de lo esperado, o que falten muestras de ese par.
    refuted: tuple[Pair, ...]
    #: Ocurrió y nadie lo predijo. Es la cubeta interesante: son los pares a los
    #: que hay que mirarles las features (`ARQUITECTURA.md` §4.8).
    unforeseen: tuple[Pair, ...]
    #: Predicho pero con al menos una letra fuera del alcance de la fase.
    not_evaluable: tuple[Pair, ...]
    #: Cuántas confusiones hacen falta para considerar que un par "ocurrió".
    min_confusions: int


def hypothesis_contrast(
    predicted: frozenset[Pair],
    confusion: Mapping[Pair, int],
    evaluated: frozenset[str],
    min_confusions: int = 1,
) -> Contrast:
    """Cruza las predicciones del glosario con la matriz observada.

    `evaluated` son las etiquetas que de verdad tuvieron muestras en la
    evaluación. Un par predicho cuyas dos letras estén en el alcance pero de las
    que no se grabó nada tampoco es refutable, y por eso el criterio es la
    presencia real y no la pertenencia a `PHASE2_LABELS`.
    """
    observados = {
        par: cuenta
        for par, cuenta in most_confused(confusion, limit=len(confusion) or 1)
        if cuenta >= min_confusions
    }

    confirmed: list[Pair] = []
    refuted: list[Pair] = []
    not_evaluable: list[Pair] = []
    for izquierda, derecha in predicted:
        par = as_pair(izquierda, derecha)
        if par[0] not in evaluated or par[1] not in evaluated:
            not_evaluable.append(par)
        elif par in observados:
            confirmed.append(par)
        else:
            refuted.append(par)

    predichos_normalizados = {as_pair(a, b) for a, b in predicted}
    unforeseen = [par for par in observados if par not in predichos_normalizados]

    return Contrast(
        confirmed=tuple(sorted(confirmed)),
        refuted=tuple(sorted(refuted)),
        unforeseen=tuple(sorted(unforeseen)),
        not_evaluable=tuple(sorted(not_evaluable)),
        min_confusions=min_confusions,
    )


# --------------------------------------------------------------------------- #
# Barrido de calibración
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Axis:
    """Un eje de la rejilla: qué campo de `config.yaml` y con qué valores.

    El campo se nombra por su ruta punteada (`static_knn.max_distance`), la misma
    con la que aparece en el archivo, para que el reporte se pueda leer al lado de
    la configuración sin traducir nombres.
    """

    path: str
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class GridPoint:
    """Un punto de la rejilla y lo que midió."""

    overrides: tuple[tuple[str, Any], ...]
    accuracy: float
    macro_accuracy: float
    unknown_rate: float

    def override(self, path: str) -> Any:
        return dict(self.overrides)[path]


#: Ejes que **por construcción** no pueden mover ninguna métrica del camino
#: estático. No es una observación sobre un dataset, es una propiedad del código:
#:
#: - `features.trajectory_weight` solo pondera el canal de trayectoria
#:   (`feature-spec.md` §3.3), y `static_knn` consume la agregación del §2.
#: - `segmentation.velocity_threshold` lo consume la máquina de estados para
#:   decidir cuándo una ventana está quieta; en la evaluación las ventanas llegan
#:   ya recortadas y etiquetadas.
#: - los dos de `dtw` gobiernan el remuestreo del canal dinámico.
#:
#: La distinción importa al escribir el reporte. Un eje plano puede serlo por dos
#: motivos muy distintos —porque no toca este camino, o porque el corpus es
#: demasiado fácil y ningún umbral llega a discriminar— y explicar el segundo con
#: el argumento del primero es decir algo falso.
STRUCTURALLY_INERT: Final[frozenset[str]] = frozenset(
    {
        "features.trajectory_weight",
        "segmentation.velocity_threshold",
        "dtw.band_radius",
        "dtw.min_source_frames",
    }
)


@dataclass(frozen=True, slots=True)
class SweepResult:
    """El barrido completo, con la lectura ya hecha.

    La clasificación de los ejes evita una conversación repetida cada vez que
    alguien abre el reporte y ve una columna constante.
    """

    protocol: Protocol
    axes: tuple[Axis, ...]
    points: tuple[GridPoint, ...]
    baseline: GridPoint
    best: GridPoint
    #: Ejes que movieron alguna métrica.
    effective: tuple[str, ...]
    #: Ejes que no movieron nada, por cualquiera de los dos motivos.
    inert: tuple[str, ...]
    #: Los inertes que lo son por construcción (`STRUCTURALLY_INERT`).
    structurally_inert: tuple[str, ...]
    #: Los inertes que **deberían** haber podido mover algo y no lo hicieron. No
    #: son un hecho sobre el código sino sobre este corpus: o los valores
    #: barridos caen todos del mismo lado del umbral útil, o las clases se
    #: separan tanto que ningún umbral razonable llega a discriminar.
    flat_here: tuple[str, ...]
    #: Ejes de `STRUCTURALLY_INERT` que sí movieron algo. Debería estar siempre
    #: vacío; si no lo está, o el camino estático cambió o hay un error, y en
    #: cualquiera de los dos casos el reporte tiene que gritarlo.
    unexpected: tuple[str, ...]


def _split_path(path: str) -> tuple[str, str]:
    section, _, field = path.partition(".")
    if not section or not field or "." in field:
        msg = f"ruta de configuración inválida: {path!r}. Se espera 'seccion.campo'"
        raise ValueError(msg)
    return section, field


def _check_paths(config: Config, axes: tuple[Axis, ...]) -> None:
    """Valida las rutas **antes** de empezar, no a las dos horas de barrido."""
    for axis in axes:
        section, field = _split_path(axis.path)
        seccion = getattr(config, section, None)
        if seccion is None or not hasattr(seccion, field):
            msg = (
                f"el eje {axis.path!r} no existe en config.yaml: no hay "
                f"'{field}' dentro de '{section}'"
            )
            raise ValueError(msg)
        if not axis.values:
            raise ValueError(f"el eje {axis.path!r} no tiene valores que barrer")


def _with_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """Copia la configuración con los campos del punto de rejilla cambiados.

    Va por `model_dump` y `model_validate` a propósito: así el punto de rejilla
    pasa por la misma validación que `config.yaml`, y un valor fuera de rango
    revienta aquí en vez de producir una fila absurda en el reporte.
    """
    crudo: dict[str, Any] = config.model_dump(mode="json")
    for path, value in overrides.items():
        section, field = _split_path(path)
        crudo[section][field] = value
    return Config.model_validate(crudo)


#: Campos que cambian el vector de features y obligan a re-extraer. El resto del
#: barrido solo mueve umbrales, que se aplican sobre rankings ya calculados.
_EXTRACTION_PATHS: Final = frozenset(
    {
        "smoothing.alpha",
        "features.trajectory_weight",
        "dtw.min_source_frames",
    }
)


def sweep(
    samples: AbcSequence[Sample],
    config: Config,
    axes: tuple[Axis, ...],
    protocol: Protocol,
    labels: tuple[str, ...] = PHASE2_LABELS,
) -> SweepResult:
    """Recorre el producto cartesiano de los ejes y mide cada punto.

    El coste está dominado por la extracción de features, así que se cachea por
    los campos que de verdad la cambian, y las distancias se cachean además por
    métrica. Un barrido de cientos de puntos acaba costando unas pocas
    extracciones y un montón de comparaciones baratas.

    **La caché no es una optimización inocente: es también la medición.** Barrer
    `features.trajectory_weight` re-extrae de verdad —está en `_EXTRACTION_PATHS`—
    y que el resultado salga idéntico es la prueba empírica de que ese eje no
    toca el camino estático, no una suposición del que escribió el reporte.
    """
    _check_paths(config, axes)

    combinaciones = tuple(itertools.product(*(axis.values for axis in axes)))
    rutas = tuple(axis.path for axis in axes)

    datasets: dict[tuple[Any, ...], Dataset] = {}
    distancias: dict[tuple[Any, ...], tuple[FoldDistances, ...]] = {}

    points: list[GridPoint] = []
    for combinacion in combinaciones:
        overrides = dict(zip(rutas, combinacion, strict=True))
        punto_config = _with_overrides(config, overrides)

        clave_extraccion = tuple(
            _leer(punto_config, ruta) for ruta in sorted(_EXTRACTION_PATHS)
        )
        if clave_extraccion not in datasets:
            datasets[clave_extraccion] = observe(samples, punto_config, labels)
        dataset = datasets[clave_extraccion]

        metric = punto_config.static_knn.metric
        clave_distancias = (*clave_extraccion, metric.value, protocol.value)
        if clave_distancias not in distancias:
            distancias[clave_distancias] = precompute(
                dataset, build_folds(dataset, protocol), metric
            )

        report = score(
            distancias[clave_distancias], Thresholds.from_config(punto_config)
        )
        points.append(
            GridPoint(
                overrides=tuple(sorted(overrides.items())),
                accuracy=report.accuracy,
                macro_accuracy=report.macro_accuracy,
                unknown_rate=report.unknown_rate,
            )
        )

    baseline_overrides = tuple(
        sorted((axis.path, _leer(config, axis.path)) for axis in axes)
    )
    baseline = next(
        (p for p in points if p.overrides == baseline_overrides),
        _measure_baseline(samples, config, axes, protocol, labels),
    )

    # El mejor punto se elige por accuracy y se desempata por menos rechazos y
    # luego por los overrides, en ese orden. Sin el desempate por overrides, dos
    # ejecuciones con puntos empatados podrían elegir distinto según el orden del
    # diccionario, y el archivo de calibración dejaría de ser reproducible.
    best = min(
        points,
        key=lambda p: (-p.accuracy, p.unknown_rate, [str(v) for _, v in p.overrides]),
    )

    effective: list[str] = []
    inert: list[str] = []
    for axis in axes:
        if _moves_something(points, axis.path):
            effective.append(axis.path)
        else:
            inert.append(axis.path)

    return SweepResult(
        protocol=protocol,
        axes=axes,
        points=tuple(points),
        baseline=baseline,
        best=best,
        effective=tuple(effective),
        inert=tuple(inert),
        structurally_inert=tuple(p for p in inert if p in STRUCTURALLY_INERT),
        flat_here=tuple(p for p in inert if p not in STRUCTURALLY_INERT),
        unexpected=tuple(p for p in effective if p in STRUCTURALLY_INERT),
    )


def _leer(config: Config, path: str) -> Any:
    section, field = _split_path(path)
    valor = getattr(getattr(config, section), field)
    return valor.value if isinstance(valor, StrEnum) else valor


def _measure_baseline(
    samples: AbcSequence[Sample],
    config: Config,
    axes: tuple[Axis, ...],
    protocol: Protocol,
    labels: tuple[str, ...],
) -> GridPoint:
    """La configuración tal cual está en `config.yaml`, aunque no caiga en la
    rejilla. Es contra lo que se compara la mejora, así que tiene que existir
    siempre."""
    dataset = observe(samples, config, labels)
    report = score(
        precompute(dataset, build_folds(dataset, protocol), config.static_knn.metric),
        Thresholds.from_config(config),
    )
    return GridPoint(
        overrides=tuple(sorted((axis.path, _leer(config, axis.path)) for axis in axes)),
        accuracy=report.accuracy,
        macro_accuracy=report.macro_accuracy,
        unknown_rate=report.unknown_rate,
    )


def _moves_something(points: AbcSequence[GridPoint], path: str) -> bool:
    """Si cambiar este eje cambió alguna métrica, con todo lo demás fijo.

    Compara por grupos: dos puntos que solo difieren en `path`. Mirar la varianza
    global de la columna no serviría —otro eje podría estar moviendo el número— y
    diría que un eje inerte es efectivo.
    """
    grupos: dict[tuple[tuple[str, Any], ...], set[tuple[float, float, float]]] = {}
    for point in points:
        resto = tuple((k, v) for k, v in point.overrides if k != path)
        grupos.setdefault(resto, set()).add(
            (point.accuracy, point.macro_accuracy, point.unknown_rate)
        )
    return any(len(metricas) > 1 for metricas in grupos.values())
