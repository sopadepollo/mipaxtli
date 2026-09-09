"""`make eval` — reporte de la Fase 2: métricas, barrido y contraste de hipótesis.

Un solo comando produce tres archivos en `data/models/eval/`:

- `reporte-fase2.md` — para leer. Accuracy global y por letra, matriz de
  confusión, pares más confundidos, contraste de hipótesis, barrido y diagnóstico.
- `resultados-fase2.json` — para que lo lea otro programa.
- `calibracion-fase2.json` — los umbrales que el barrido recomienda, **con la
  huella del dataset contra el que se ajustaron**.

## Nada de esto lleva marca de tiempo

Y es deliberado. El criterio de aceptación de la fase es que el reporte sea
reproducible, y un `now()` lo rompería en el primer segundo. Lo que identifica una
ejecución es la huella del corpus más el commit, que son propiedades de la entrada
y no del momento en que se apretó *enter*.

## El contraste de hipótesis

La columna `confundible_con` de `docs/glosario-lsm.md` se llenó leyendo *Manos con
voz*, **sin datos y antes de grabar nada**. Eso la convierte en una hipótesis
falsable, y esta es la única parte del proyecto donde se pone a prueba. El reporte
la cruza con la matriz real y separa cuatro casos; el cuarto —los pares que tocan
una letra dinámica y que la Fase 2 no puede evaluar— existe porque son trece de los
veintiséis, y meterlos entre los refutados diría que el glosario se equivocó
cuando lo que pasa es que todavía no se ha medido.

Este módulo vive en `cli/` porque lee el glosario y escribe archivos. Todo lo que
decide qué significa un número está en `lsm.evaluation`, que se testea sin disco.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from lsm.classifiers.static_knn import Thresholds
from lsm.config import Config, load_config
from lsm.evaluation import (
    EXCLUDED_LABELS,
    PHASE2_LABELS,
    Axis,
    Contrast,
    Dataset,
    InsufficientFoldsError,
    Observation,
    Pair,
    Protocol,
    Report,
    SweepResult,
    as_pair,
    build_folds,
    hypothesis_contrast,
    most_confused,
    observe,
    precompute,
    score,
    sweep,
)
from lsm.io.corpus import (
    SYNTHETIC_REPETITIONS,
    SYNTHETIC_SIGNERS,
    SYNTHETIC_WARNING,
    Corpus,
    CorpusError,
    load_corpus,
)
from lsm.io.glossary import DEFAULT_GLOSSARY, parse_confundible, read_letter_table
from lsm.types import UNKNOWN_LABEL

DEFAULT_OUTPUT = Path("data/models/eval")

#: Criterio de aceptación de la Fase 2 (`ARQUITECTURA.md` §5).
TARGET_ACCURACY: Final = 0.90

#: Cuántos pares confundidos se listan.
TOP_PAIRS: Final = 15


# --------------------------------------------------------------------------- #
# La rejilla de calibración
# --------------------------------------------------------------------------- #
#
# `features.trajectory_weight` y `segmentation.velocity_threshold` se barren
# aunque **no puedan** mover el accuracy del camino estático: el primero solo
# pondera el canal de trayectoria, que `static_knn` no usa, y el segundo solo lo
# consume la máquina de estados, que sobre muestras ya recortadas no interviene.
#
# Se barren igualmente por dos motivos. El primero es que la inercia sea una
# medición y no una afirmación: el barrido re-extrae features de verdad al mover
# `trajectory_weight`, y que el resultado salga idéntico lo demuestra. El segundo
# es que quien lea el reporte dentro de seis meses no tenga que volver a
# preguntárselo.
#
# Lo que sí los calibra es la sección de diagnóstico empírico, que enseña las
# distribuciones reales de velocidad, σ y longitud de arco por clase.

_FULL_GRID: Final[tuple[Axis, ...]] = (
    Axis(path="static_knn.metric", values=("euclidean", "cosine")),
    Axis(path="static_knn.max_distance", values=(0.25, 0.5, 1.0, 2.0, 4.0)),
    Axis(path="static_knn.min_margin", values=(0.50, 0.55, 0.60, 0.70, 0.80)),
    Axis(path="quality.max_dispersion", values=(0.02, 0.04, 0.08, 0.16, 0.32)),
    Axis(path="features.trajectory_weight", values=(0.0, 4.0, 20.0)),
    Axis(path="segmentation.velocity_threshold", values=(0.01, 0.02, 0.05)),
)

_SMALL_GRID: Final[tuple[Axis, ...]] = (
    Axis(path="static_knn.metric", values=("euclidean", "cosine")),
    Axis(path="static_knn.max_distance", values=(0.25, 2.0)),
    Axis(path="static_knn.min_margin", values=(0.50, 0.80)),
    Axis(path="quality.max_dispersion", values=(0.02, 0.32)),
    Axis(path="features.trajectory_weight", values=(0.0, 20.0)),
    Axis(path="segmentation.velocity_threshold", values=(0.01, 0.05)),
)

GRIDS: Final[dict[str, tuple[Axis, ...]]] = {
    "completo": _FULL_GRID,
    "minimo": _SMALL_GRID,
    "ninguno": (),
}


# --------------------------------------------------------------------------- #
# Glosario: las predicciones teóricas
# --------------------------------------------------------------------------- #


def predicted_pairs(glossary: Path) -> frozenset[Pair]:
    """Los pares `confundible_con` del glosario, normalizados y sin duplicar.

    La tabla anota la relación en las dos filas —A dice que se confunde con E, y E
    con A— y aquí las dos se colapsan al mismo par ordenado. Un par que solo
    apareciera en una fila también entra: la asimetría es un descuido al llenar la
    tabla, no una afirmación de que la confusión ocurra en un solo sentido.
    """
    pares: set[Pair] = set()
    for fila in read_letter_table(glossary):
        origen = fila["label"]
        for destino in parse_confundible(fila["confundible_con"]):
            if origen != destino:
                pares.add(as_pair(origen, destino))
    return frozenset(pares)


# --------------------------------------------------------------------------- #
# Formato
# --------------------------------------------------------------------------- #


def abbreviate(label: str) -> str:
    """Etiqueta a como mucho cuatro caracteres, para que la matriz quepa a lo ancho.

    Las 21 estáticas son de una sola letra; `NONE` y `UNKNOWN` son las únicas que
    hay que acortar. Cuando la Fase 5 sume `DOBLE_L` y compañía, esto las deja en
    `DOBL`, que colisiona — y por eso la función devuelve algo distinto para cada
    una en vez de truncar a ciegas.
    """
    especiales = {
        "NONE": "NON",
        UNKNOWN_LABEL: "UNK",
        "DOBLE_L": "LL",
        "DOBLE_R": "RR",
        "ENIE": "Ñ",
    }
    return especiales.get(label, label[:4])


def render_confusion(report: Report, labels: Sequence[str]) -> str:
    """La matriz como bloque de texto alineado.

    En texto plano y no en tabla de Markdown a propósito: 22 columnas hacen una
    tabla de Markdown ilegible en cualquier ancho, y esto se lee en un terminal
    tanto como en un navegador. La columna `UNK` va al final y separada porque un
    rechazo no es una confusión con otra letra.
    """
    columnas = [*labels, UNKNOWN_LABEL]
    ancho = max(4, max(len(abbreviate(c)) for c in columnas) + 1)

    cabecera = " " * (ancho + 1) + "".join(
        f"{abbreviate(c):>{ancho}}" for c in columnas
    )
    lineas = [cabecera, " " * (ancho + 1) + "-" * (ancho * len(columnas))]
    for verdadera in labels:
        celdas = []
        for predicha in columnas:
            cuenta = report.confusion.get((verdadera, predicha), 0)
            celdas.append(f"{cuenta if cuenta else '.':>{ancho}}")
        lineas.append(f"{abbreviate(verdadera):>{ancho}}|" + "".join(celdas))
    return "\n".join(lineas)


def _tabla(cabeceras: Sequence[str], filas: Sequence[Sequence[str]]) -> str:
    salida = ["| " + " | ".join(cabeceras) + " |"]
    salida.append("|" + "|".join("---" for _ in cabeceras) + "|")
    for fila in filas:
        salida.append("| " + " | ".join(fila) + " |")
    return "\n".join(salida)


def _lista_de_pares(pares: Sequence[Pair], vacio: str) -> str:
    if not pares:
        return f"_{vacio}_"
    return "\n".join(f"- `{a}` · `{b}`" for a, b in pares)


# --------------------------------------------------------------------------- #
# Secciones del reporte
# --------------------------------------------------------------------------- #


def _cabecera(corpus: Corpus, protocol: Protocol, folds: int, config: Config) -> str:
    p = corpus.provenance
    aviso = f"> **{SYNTHETIC_WARNING}**\n\n" if p.synthetic else ""
    return (
        "# Reporte de evaluación — Fase 2\n\n"
        f"{aviso}"
        + _tabla(
            ["campo", "valor"],
            [
                ["protocolo", f"`{protocol.value}` ({folds} folds)"],
                ["corpus", f"`{p.source}`"],
                ["huella del dataset", f"`{p.fingerprint}`"],
                ["muestras", str(p.sample_count)],
                ["firmantes", ", ".join(f"`{s}`" for s in p.signers)],
                ["sesiones", str(len(p.sessions))],
                ["commit", f"`{p.git_commit}`" if p.git_commit else "_no disponible_"],
                ["feature_spec_version", str(p.to_json()["feature_spec_version"])],
                ["métrica", f"`{config.static_knn.metric.value}`"],
                ["max_distance", f"{config.static_knn.max_distance}"],
                ["min_margin", f"{config.static_knn.min_margin}"],
                ["max_dispersion", f"{config.quality.max_dispersion}"],
            ],
        )
        + "\n"
    )


def _seccion_alcance(dataset: Dataset) -> str:
    descartadas = sum(dataset.excluded.values())
    rechazadas = sum(dataset.rejected.values())
    detalle_rechazo = (
        ", ".join(f"`{k}`={v}" for k, v in sorted(dataset.rejected.items()))
        or "ninguna"
    )
    faltantes = sorted(set(PHASE2_LABELS) - set(dataset.labels))

    texto = [
        "## Alcance",
        "",
        "La Fase 2 evalúa **21 letras estáticas más la clase negativa `NONE`**: "
        "22 clases.",
        "",
        "Las **8 letras dinámicas quedan explícitamente excluidas hasta la Fase 5**: "
        + ", ".join(f"`{label}`" for label in sorted(EXCLUDED_LABELS))
        + ". `static_knn` promedia los frames de la secuencia; el promedio de una "
        "Z no es una configuración de mano sino una mancha, y aterrizaría sobre "
        "alguna estática ensuciando su fila de la matriz.",
        "",
        "Por el mismo motivo se descartan las muestras de `NONE` **grabadas en "
        "modo dinámico** —transiciones y saludos—, que no salen por su etiqueta "
        "sino por su `kind`. La clase negativa se graba de las dos formas a "
        "propósito, pero `static_knn` le da un único centroide y promediar un "
        "saludo no da ninguna configuración de mano. Esperan a la Fase 5 igual que "
        "las ocho letras. Ver "
        "`docs/adr/0010-la-clase-negativa-y-el-modo-de-grabacion.md`.",
        "",
        _tabla(
            ["concepto", "muestras"],
            [
                ["evaluadas", str(len(dataset.observations))],
                ["descartadas por dinámicas", str(descartadas)],
                [
                    "descartadas por ser `NONE` dinámica",
                    str(dataset.dynamic_negatives),
                ],
                ["rechazadas por la extracción", f"{rechazadas} ({detalle_rechazo})"],
            ],
        ),
    ]
    if faltantes:
        texto += [
            "",
            "**Clases del alcance sin ninguna muestra**: "
            + ", ".join(f"`{label}`" for label in faltantes)
            + ". No aparecen en la matriz y el accuracy no las cuenta; el modelo "
            "no puede reconocerlas.",
        ]
    return "\n".join(texto) + "\n"


def _seccion_global(report: Report) -> str:
    cumple = "**CUMPLE**" if report.accuracy >= TARGET_ACCURACY else "**NO CUMPLE**"
    return (
        "## Accuracy global\n\n"
        + _tabla(
            ["métrica", "valor"],
            [
                ["accuracy", f"{report.accuracy:.4f}"],
                ["accuracy macro (media por letra)", f"{report.macro_accuracy:.4f}"],
                ["tasa de UNKNOWN", f"{report.unknown_rate:.4f}"],
                ["muestras evaluadas", str(report.total)],
                ["aciertos", str(report.correct)],
                [f"criterio de la fase (≥ {TARGET_ACCURACY:.0%})", cumple],
            ],
        )
        + "\n\nUn `UNKNOWN` cuenta como fallo en el accuracy. Es lo severo y es lo "
        "correcto: una configuración que se calla siempre no reconoce ninguna "
        "letra. La tasa de rechazo va aparte para poder leer las dos juntas.\n\n"
        "El **accuracy macro** es el que hay que mirar si el dataset quedó "
        "desbalanceado: con muchas muestras de `NONE` y pocas por letra, el "
        "accuracy global se sostiene acertando solo la clase negativa.\n"
    )


def _seccion_por_letra(report: Report) -> str:
    filas = [
        [
            f"`{label}`",
            f"{s.accuracy:.4f}",
            f"{s.correct}/{s.total}",
            str(s.unknown),
        ]
        for label, s in sorted(
            report.per_label.items(), key=lambda item: (item[1].accuracy, item[0])
        )
    ]
    return (
        "## Accuracy por letra\n\n"
        "Ordenado de peor a mejor: arriba está lo que hay que arreglar.\n\n"
        + _tabla(["letra", "accuracy", "aciertos", "UNKNOWN"], filas)
        + "\n"
    )


def _seccion_matriz(report: Report, labels: Sequence[str]) -> str:
    return (
        "## Matriz de confusión\n\n"
        "Filas: etiqueta verdadera. Columnas: predicha. `UNK` es rechazo. "
        "Solo clases estáticas y `NONE`.\n\n"
        "```\n" + render_confusion(report, labels) + "\n```\n"
    )


def _seccion_pares(pares: Sequence[tuple[Pair, int]]) -> str:
    if not pares:
        cuerpo = "_Ningún par de letras se confundió entre sí._"
    else:
        cuerpo = _tabla(
            ["par", "confusiones (ambos sentidos)"],
            [[f"`{a}` · `{b}`", str(n)] for (a, b), n in pares],
        )
    return (
        "## Pares más confundidos\n\n"
        "Suma de las dos direcciones. `UNKNOWN` queda fuera: un rechazo no es una "
        "confusión **entre dos letras**, y como suele ser la celda más grande de "
        "la fila taparía justo lo que esta lista existe para enseñar.\n\n"
        + cuerpo
        + "\n"
    )


def _seccion_contraste(contrast: Contrast, predichos: int, synthetic: bool) -> str:
    aviso_sintetico = (
        [
            "> ⚠️ **Sobre corpus sintético, esta sección no dice nada del "
            "glosario.** Las manos de `lsm.synthetic` se derivan de una rejilla "
            "de flexiones, no de LSM: que un par predicho aparezca aquí como "
            "confirmado es una coincidencia entre esa rejilla y la anatomía real, "
            "y que aparezca como refutado solo dice que esas dos manos "
            "artificiales no se parecen. La hipótesis se pone a prueba con "
            "grabaciones.",
            "",
        ]
        if synthetic
        else []
    )
    return "\n".join(
        [
            "## Contraste de hipótesis",
            "",
            *aviso_sintetico,
            "La columna `confundible_con` de `docs/glosario-lsm.md` se llenó "
            "leyendo *Manos con voz*, **antes de grabar nada y sin mirar un solo "
            "dato**. Es una hipótesis falsable y esto es ponerla a prueba.",
            "",
            f"Pares predichos en el glosario: **{predichos}**. "
            f"Umbral para considerar que un par «se confundió»: "
            f"**{contrast.min_confusions}** confusión(es).",
            "",
            "### Predichos que sí se confundieron",
            "",
            "La intuición sobre la configuración manual se tradujo en una "
            "confusión real del clasificador.",
            "",
            _lista_de_pares(contrast.confirmed, "ninguno"),
            "",
            "### Predichos que NO se confundieron",
            "",
            "Ambas letras se evaluaron y el clasificador las separó. O las "
            "features distinguen mejor de lo que la descripción sugería, o falta "
            "muestra de ese par para que la confusión aparezca.",
            "",
            _lista_de_pares(contrast.refuted, "ninguno: todo lo predicho ocurrió"),
            "",
            "### Confundidos sin estar predichos",
            "",
            "La cubeta interesante: pares que nadie anticipó leyendo el glosario. "
            "Son los candidatos a features específicas —ángulos entre falanges, "
            "distancias pulgar-dedos— antes que a cambiar de modelo "
            "(`ARQUITECTURA.md` §4.8). Valdría la pena anotarlos de vuelta en el "
            "glosario.",
            "",
            _lista_de_pares(contrast.unforeseen, "ninguno"),
            "",
            "### Predichos pero no evaluables en la Fase 2",
            "",
            "Al menos una de las dos letras es dinámica o no tuvo muestras. **No "
            "son refutaciones**: son predicciones que todavía no se han medido, y "
            "esperan a la Fase 5.",
            "",
            _lista_de_pares(contrast.not_evaluable, "ninguno"),
            "",
        ]
    )


def _seccion_barrido(result: SweepResult | None) -> str:
    if result is None:
        return "## Barrido de calibración\n\n_No se ejecutó (`--barrido ninguno`)._\n"

    filas_ejes = []
    for axis in result.axes:
        if axis.path in result.effective:
            estado = "**efectivo**"
        elif axis.path in result.structurally_inert:
            estado = "inerte por construcción"
        else:
            estado = "plano en este corpus"
        valores = ", ".join(f"`{v}`" for v in axis.values)
        mejor = result.best.override(axis.path)
        filas_ejes.append([f"`{axis.path}`", valores, estado, f"`{mejor}`"])

    notas = []
    if result.structurally_inert:
        notas.append(
            "**Inertes por construcción**: "
            + ", ".join(f"`{p}`" for p in result.structurally_inert)
            + ". No es que el barrido no los haya movido: los movió y no cambió "
            "nada, porque no participan en el camino estático. "
            "`features.trajectory_weight` solo pondera el canal de trayectoria "
            "(`feature-spec.md` §3.3) y `static_knn` consume la agregación del §2; "
            "`segmentation.velocity_threshold` lo usa la máquina de estados para "
            "decidir cuándo una ventana está quieta, y aquí las ventanas llegan "
            "ya recortadas. Que salgan planos está **medido**, no supuesto: el "
            "barrido vuelve a extraer las features de verdad al cambiar "
            "`trajectory_weight`. Se calibran con la sección siguiente."
        )
    if result.flat_here:
        notas.append(
            "**Planos en este corpus**: "
            + ", ".join(f"`{p}`" for p in result.flat_here)
            + ". Estos sí afectan a la decisión, y aun así ningún valor de los "
            "barridos cambió una sola métrica. Eso dice algo del **corpus**, no "
            "del código: o los valores probados caen todos del mismo lado del "
            "umbral útil, o las clases se separan tanto que ninguna puerta llega "
            "a discriminar. Mirar la sección de diagnóstico antes de concluir que "
            "el valor actual está bien elegido."
        )
    if result.unexpected:
        notas.append(
            "> ⚠️ **"
            + ", ".join(f"`{p}`" for p in result.unexpected)
            + " movió alguna métrica y no debería.** O el camino estático cambió "
            "—y entonces hay que actualizar `evaluation.STRUCTURALLY_INERT` y "
            "probablemente un ADR— o hay un error. No usar este barrido para "
            "calibrar nada hasta aclararlo."
        )
    explicacion_inertes = ("\n\n" + "\n\n".join(notas)) if notas else ""

    mejora = result.best.accuracy - result.baseline.accuracy
    return "\n".join(
        [
            "## Barrido de calibración",
            "",
            f"Puntos de la rejilla: **{len(result.points)}**. "
            f"Protocolo: `{result.protocol.value}`.",
            "",
            _tabla(["eje", "valores barridos", "efecto", "mejor valor"], filas_ejes),
            explicacion_inertes,
            "",
            "### Configuración actual contra la mejor encontrada",
            "",
            _tabla(
                ["", "accuracy", "accuracy macro", "tasa UNKNOWN"],
                [
                    [
                        "`config.yaml` actual",
                        f"{result.baseline.accuracy:.4f}",
                        f"{result.baseline.macro_accuracy:.4f}",
                        f"{result.baseline.unknown_rate:.4f}",
                    ],
                    [
                        "mejor de la rejilla",
                        f"{result.best.accuracy:.4f}",
                        f"{result.best.macro_accuracy:.4f}",
                        f"{result.best.unknown_rate:.4f}",
                    ],
                    ["diferencia", f"{mejora:+.4f}", "", ""],
                ],
            ),
            "",
            "Los valores recomendados están en `calibracion-fase2.json`, **junto "
            "con la huella del dataset contra el que se ajustaron**. Copiarlos a "
            "`config.yaml` sin comprobar que esa huella coincide con el dataset "
            "actual es aplicar una calibración de otro corpus.",
            "",
        ]
    )


def _percentil(valores: Sequence[float], fraccion: float) -> float:
    """Percentil por interpolación lineal. Determinista y sin dependencias."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]
    posicion = fraccion * (len(ordenados) - 1)
    bajo = int(posicion)
    if bajo >= len(ordenados) - 1:
        return ordenados[-1]
    return ordenados[bajo] + (posicion - bajo) * (ordenados[bajo + 1] - ordenados[bajo])


def _seccion_diagnostico(dataset: Dataset, config: Config) -> str:
    """Las distribuciones que sí calibran los umbrales que el accuracy no mueve."""
    observaciones = dataset.observations
    sigmas = [o.dispersion for o in observaciones]
    velocidades = [o.max_velocity for o in observaciones]
    arcos = [o.arc_length for o in observaciones]

    filas_globales = [
        [
            "σ (dispersión)",
            "`quality.max_dispersion`",
            f"{config.quality.max_dispersion}",
            f"{_percentil(sigmas, 0.5):.4f}",
            f"{_percentil(sigmas, 0.95):.4f}",
            f"{max(sigmas, default=0.0):.4f}",
        ],
        [
            "velocidad máxima",
            "`segmentation.velocity_threshold`",
            f"{config.segmentation.velocity_threshold}",
            f"{_percentil(velocidades, 0.5):.4f}",
            f"{_percentil(velocidades, 0.95):.4f}",
            f"{max(velocidades, default=0.0):.4f}",
        ],
        [
            "longitud de arco de τ",
            "`capture.min_trajectory_arc`",
            f"{config.capture.min_trajectory_arc}",
            f"{_percentil(arcos, 0.5):.4f}",
            f"{_percentil(arcos, 0.95):.4f}",
            f"{max(arcos, default=0.0):.4f}",
        ],
    ]

    por_clase: dict[str, list[Observation]] = {}
    for o in observaciones:
        por_clase.setdefault(o.label, []).append(o)
    filas_clase = [
        [
            f"`{label}`",
            f"{_percentil([x.dispersion for x in grupo], 0.5):.4f}",
            f"{max(x.dispersion for x in grupo):.4f}",
            f"{_percentil([x.max_velocity for x in grupo], 0.5):.4f}",
            f"{_percentil([x.arc_length for x in grupo], 0.5):.4f}",
        ]
        for label, grupo in sorted(por_clase.items())
    ]

    return "\n".join(
        [
            "## Diagnóstico empírico de umbrales",
            "",
            "Esta sección es lo que de verdad calibra los umbrales que el accuracy "
            "no mueve. `velocity_threshold` y `min_trajectory_arc` no deciden nada "
            "sobre muestras ya recortadas, pero sí sobre el video en vivo de la "
            "Fase 3 y sobre qué se acepta al grabar — y hasta ahora sus valores "
            "eran, literalmente, «un punto de partida razonado, no medido».",
            "",
            "### Distribución global",
            "",
            _tabla(
                [
                    "magnitud",
                    "umbral que la usa",
                    "valor actual",
                    "mediana",
                    "p95",
                    "máximo",
                ],
                filas_globales,
            ),
            "",
            "Cómo leerlo: un umbral por debajo del p95 de una magnitud rechazará "
            "al menos el 5% de lo que hoy se acepta. Para `max_dispersion` eso son "
            "ventanas que pasarían a `UNKNOWN`; para `min_trajectory_arc`, "
            "muestras dinámicas que la captura dejaría de guardar.",
            "",
            "### Por clase",
            "",
            _tabla(
                ["clase", "σ mediana", "σ máxima", "velocidad mediana", "arco mediano"],
                filas_clase,
            ),
            "",
            "Una clase con σ mucho mayor que el resto se grabó peor, no es más "
            "difícil. Vale la pena mirar esas muestras antes de tocar el modelo.",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
# Ensamblado y escritura
# --------------------------------------------------------------------------- #


def build_report(
    corpus: Corpus,
    dataset: Dataset,
    config: Config,
    protocol: Protocol,
    folds: int,
    report: Report,
    pares: Sequence[tuple[Pair, int]],
    contrast: Contrast,
    predichos: int,
    barrido: SweepResult | None,
) -> str:
    return "\n".join(
        [
            _cabecera(corpus, protocol, folds, config),
            _seccion_alcance(dataset),
            _seccion_global(report),
            _seccion_por_letra(report),
            _seccion_matriz(report, dataset.labels),
            _seccion_pares(pares),
            _seccion_contraste(contrast, predichos, corpus.provenance.synthetic),
            _seccion_barrido(barrido),
            _seccion_diagnostico(dataset, config),
        ]
    )


def _resultados_json(
    corpus: Corpus,
    dataset: Dataset,
    protocol: Protocol,
    report: Report,
    pares: Sequence[tuple[Pair, int]],
    contrast: Contrast,
) -> dict[str, Any]:
    return {
        "dataset": corpus.provenance.to_json(),
        "protocol": protocol.value,
        "labels": list(dataset.labels),
        "excluded_labels": sorted(EXCLUDED_LABELS),
        "accuracy": report.accuracy,
        "macro_accuracy": report.macro_accuracy,
        "unknown_rate": report.unknown_rate,
        "total": report.total,
        "correct": report.correct,
        "per_label": {
            label: {
                "accuracy": s.accuracy,
                "total": s.total,
                "correct": s.correct,
                "unknown": s.unknown,
            }
            for label, s in sorted(report.per_label.items())
        },
        "confusion": [
            {"truth": verdadera, "predicted": predicha, "count": cuenta}
            for (verdadera, predicha), cuenta in sorted(report.confusion.items())
        ],
        "most_confused": [
            {"pair": list(par), "count": cuenta} for par, cuenta in pares
        ],
        "hypothesis_contrast": {
            "min_confusions": contrast.min_confusions,
            "confirmed": [list(p) for p in contrast.confirmed],
            "refuted": [list(p) for p in contrast.refuted],
            "unforeseen": [list(p) for p in contrast.unforeseen],
            "not_evaluable": [list(p) for p in contrast.not_evaluable],
        },
    }


def _calibracion_json(
    corpus: Corpus, protocol: Protocol, result: SweepResult
) -> dict[str, Any]:
    """Los umbrales recomendados **y** contra qué dataset se ajustaron.

    La huella no es decorativa: es lo que permite responder «¿estos valores son de
    este corpus?» sin acordarse. Un archivo de calibración cuya huella no coincida
    con el dataset actual está describiendo otra cosa.
    """

    def punto(p: Any) -> dict[str, Any]:
        return {
            "overrides": dict(p.overrides),
            "accuracy": p.accuracy,
            "macro_accuracy": p.macro_accuracy,
            "unknown_rate": p.unknown_rate,
        }

    return {
        "schema_version": 1,
        "dataset": corpus.provenance.to_json(),
        "protocol": protocol.value,
        "axes": [
            {"path": axis.path, "values": list(axis.values)} for axis in result.axes
        ],
        "effective": list(result.effective),
        "inert": list(result.inert),
        "structurally_inert": list(result.structurally_inert),
        "flat_here": list(result.flat_here),
        "unexpected_effect": list(result.unexpected),
        "baseline": punto(result.baseline),
        "best": punto(result.best),
        "grid": [punto(p) for p in result.points],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-eval",
        description=(
            "Evalúa el clasificador estático con validación leave-one-signer-out "
            "y escribe el reporte de la Fase 2. No necesita cámara ni MediaPipe."
        ),
    )
    parser.add_argument("--raiz", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--glosario", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protocolo",
        choices=[p.value for p in Protocol],
        default=Protocol.SIGNER.value,
        help=(
            "leave-one-signer-out por defecto. leave-one-session-out es una "
            "degradación consciente para cuando solo ha grabado una persona: mide "
            "generalización entre sesiones, que es mucho menos"
        ),
    )
    parser.add_argument(
        "--barrido",
        choices=sorted(GRIDS),
        default="completo",
        help="tamaño de la rejilla de calibración",
    )
    parser.add_argument(
        "--min-confusiones",
        type=int,
        default=1,
        help="confusiones necesarias para dar por observado un par del glosario",
    )
    parser.add_argument("--sin-sintetico", action="store_true")
    parser.add_argument(
        "--repeticiones-sinteticas", type=int, default=SYNTHETIC_REPETITIONS
    )
    parser.add_argument("--firmantes-sinteticos", type=int, default=SYNTHETIC_SIGNERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    protocol = Protocol(args.protocolo)

    try:
        corpus = load_corpus(
            args.raiz,
            PHASE2_LABELS,
            allow_synthetic=not args.sin_sintetico,
            repetitions=args.repeticiones_sinteticas,
            signers=args.firmantes_sinteticos,
        )
    except CorpusError as error:
        print(f"lsm-eval: {error}")
        return 1

    if corpus.provenance.synthetic:
        print(SYNTHETIC_WARNING)
        print()

    dataset = observe(corpus.samples, config)
    if not dataset.observations:
        print("lsm-eval: no quedó ninguna muestra evaluable.")
        return 1

    try:
        folds = build_folds(dataset, protocol)
    except InsufficientFoldsError as error:
        print(f"lsm-eval: {error}")
        return 1

    report = score(
        precompute(dataset, folds, config.static_knn.metric),
        Thresholds.from_config(config),
    )
    pares = most_confused(report.confusion, limit=TOP_PAIRS)
    predichos = (
        predicted_pairs(args.glosario) if args.glosario.is_file() else frozenset()
    )
    contrast = hypothesis_contrast(
        predichos,
        report.confusion,
        evaluated=frozenset(dataset.labels),
        min_confusions=args.min_confusiones,
    )

    barrido = None
    if GRIDS[args.barrido]:
        barrido = sweep(corpus.samples, config, GRIDS[args.barrido], protocol)

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "reporte-fase2.md").write_text(
        build_report(
            corpus,
            dataset,
            config,
            protocol,
            len(folds),
            report,
            pares,
            contrast,
            len(predichos),
            barrido,
        ),
        encoding="utf-8",
    )
    _escribir_json(
        args.salida / "resultados-fase2.json",
        _resultados_json(corpus, dataset, protocol, report, pares, contrast),
    )
    if barrido is not None:
        _escribir_json(
            args.salida / "calibracion-fase2.json",
            _calibracion_json(corpus, protocol, barrido),
        )

    _resumen(corpus, report, contrast, barrido, args.salida)
    return 0


def _escribir_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resumen(
    corpus: Corpus,
    report: Report,
    contrast: Contrast,
    barrido: SweepResult | None,
    salida: Path,
) -> None:
    print(f"corpus       {corpus.provenance.source} ({corpus.provenance.short})")
    print(f"muestras     {report.total}")
    print(
        f"accuracy     {report.accuracy:.4f}   "
        f"(criterio de fase: ≥ {TARGET_ACCURACY:.0%})"
    )
    print(f"macro        {report.macro_accuracy:.4f}")
    print(f"UNKNOWN      {report.unknown_rate:.4f}")
    print(
        f"hipótesis    {len(contrast.confirmed)} confirmadas, "
        f"{len(contrast.refuted)} refutadas, "
        f"{len(contrast.unforeseen)} imprevistas, "
        f"{len(contrast.not_evaluable)} no evaluables"
    )
    if barrido is not None:
        print(
            f"barrido      {len(barrido.points)} puntos; mejor accuracy "
            f"{barrido.best.accuracy:.4f}; ejes inertes: "
            f"{', '.join(barrido.inert) or 'ninguno'}"
        )
    print(f"reporte      {salida / 'reporte-fase2.md'}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
