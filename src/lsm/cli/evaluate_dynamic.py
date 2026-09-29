"""`lsm-eval-dinamico` — reporte de la Fase 5: DTW, barrido y reproducción.

Produce en `data/models/eval/`:

- `reporte-fase5.md` — para leer.
- `resultados-fase5.json` — para que lo lea otro programa.

Cuatro secciones, cada una contestando una pregunta distinta:

1. **¿Reconoce las dinámicas?** Leave-one-signer-out con la configuración de
   `config.yaml`, puerta de distancia incluida. Por letra, no solo el promedio.
2. **¿Dónde van `w_τ` y la banda?** El barrido `trajectory_weight × band_radius`
   con la puerta abierta, y **la `Z` aparte**: el glosario advirtió desde su
   redacción que `band_radius = 6` podía quedarle corto (PENDIENTE-HUMANO F).
3. **¿Dónde va `dtw.max_distance`?** La distribución de `d₁` de los aciertos
   contra la de las muestras estáticas, que son lo que el camino dinámico tiene
   que rechazar cuando le llega por error.
4. **¿La máquina de estados recorta bien el trazo?** Cada grabación dinámica
   pasa por `run_segmentation` con modelos entrenados sin su firmante: cuántas
   salen enteras, partidas o perdidas. Y cada estática, con y sin camino
   dinámico, para ver qué cambió en ellas.

Como `lsm-eval`, sin marca de tiempo: lo que identifica una ejecución es la
huella del corpus más el commit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lsm.classifiers.dynamic_dtw import (
    DynamicDtwClassifier,
    dynamic_rows,
    label_distances,
)
from lsm.classifiers.registry import ClassifierRegistry
from lsm.classifiers.static_knn import StaticKnnClassifier, rank
from lsm.config import Config, load_config
from lsm.evaluation import (
    PHASE2_LABELS,
    PHASE5_LABELS,
    DynamicDataset,
    DynamicGridPoint,
    InsufficientFoldsError,
    Protocol,
    ReplayOutcome,
    Report,
    build_folds,
    group_templates,
    nearest_distances,
    observe_dynamic,
    precompute_dynamic,
    replay_sample,
    score_dynamic,
    sweep_dynamic,
    without_dynamic_path,
)
from lsm.io.corpus import (
    SYNTHETIC_WARNING,
    Corpus,
    CorpusError,
    load_corpus,
)
from lsm.preprocessing import Preprocessing
from lsm.segmentation import FrameThresholds
from lsm.types import UNKNOWN_LABEL, Sample, SampleKind, WindowOrigin
from lsm.vocabulary import DIRECTION_PENDING_LABELS

DEFAULT_OUTPUT = Path("data/models/eval")

#: Rejillas del barrido. La completa es la que cita el ADR 0016.
GRIDS: Final[dict[str, tuple[tuple[float, ...], tuple[int, ...]]]] = {
    "completo": ((0.0, 1.0, 2.0, 4.0, 8.0, 16.0), (2, 4, 6, 8, 12, 23)),
    "rapido": ((0.0, 4.0, 16.0), (2, 6, 23)),
}

#: Percentiles de `d₁` de los aciertos que se ofrecen como candidatos a
#: `dtw.max_distance`.
PERCENTILES: Final = (0.90, 0.95, 0.99, 1.0)


def _percentil(valores: Sequence[float], fraccion: float) -> float:
    """Percentil por el método del rango más cercano: un valor que existe."""
    if not valores:
        return math.nan
    indice = min(len(valores) - 1, max(0, math.ceil(fraccion * len(valores)) - 1))
    return sorted(valores)[indice]


def _tabla(cabeceras: Sequence[str], filas: Sequence[Sequence[str]]) -> str:
    lineas = ["| " + " | ".join(cabeceras) + " |"]
    lineas.append("|" + "|".join("---" for _ in cabeceras) + "|")
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


# --------------------------------------------------------------------------- #
# Las cuatro mediciones
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Gate:
    """Qué deja pasar una `max_distance` candidata."""

    threshold: float
    origin: str
    #: Fracción de los aciertos LOSO cuyo `d₁` pasa la puerta.
    positives_kept: float
    #: Fracción de las muestras estáticas cuyo `d₁` pasa la puerta: lo que se
    #: colaría como letra dinámica si una estática entrara al camino dinámico.
    negatives_accepted: float


def measure_gate(
    dataset: DynamicDataset,
    folds_correct: tuple[float, ...],
    negatives: Sequence[Sample],
    config: Config,
) -> tuple[tuple[float, ...], tuple[Gate, ...]]:
    """`d₁` de las estáticas contra las plantillas de producción, y candidatos.

    Las plantillas son las de todo el corpus dinámico —las que `lsm-train`
    exportaría—, porque en vivo una estática que se cuela al camino dinámico se
    compara contra esas.
    """
    plantillas = group_templates(
        dataset, range(len(dataset.observations)), config.dtw.band_radius
    )
    negativos_d1: list[float] = []
    for sample in negatives:
        rows = dynamic_rows(sample.sequence, config)
        if rows is None:
            continue
        ranking = rank(label_distances(rows, plantillas, config.dtw.band_radius))
        if ranking is not None:
            negativos_d1.append(ranking.nearest)
    negativos = tuple(sorted(negativos_d1))

    candidatos = [
        (
            _percentil(folds_correct, p),
            f"p{round(p * 100)} de d₁ en aciertos" if p < 1.0 else "máximo",
        )
        for p in PERCENTILES
    ]
    candidatos.append((config.dtw.max_distance, "config.yaml"))
    gates = tuple(
        Gate(
            threshold=umbral,
            origin=origen,
            positives_kept=(
                sum(d <= umbral for d in folds_correct) / len(folds_correct)
                if folds_correct
                else math.nan
            ),
            negatives_accepted=(
                sum(d <= umbral for d in negativos) / len(negativos)
                if negativos
                else math.nan
            ),
        )
        for umbral, origen in sorted(candidatos)
    )
    return negativos, gates


def replay_loso(
    corpus: Corpus, config: Config
) -> tuple[tuple[ReplayOutcome, ...], tuple[ReplayOutcome, ...]]:
    """Reproduce cada muestra con modelos entrenados sin su firmante.

    Devuelve (dinámicas, estáticas), las dos con el camino dinámico activo. La
    referencia sin él la da `replay_static_without_dynamic`.
    """
    firmantes = sorted({s.signer_id for s in corpus.samples})
    reposo = (
        FrameThresholds.from_config(
            config, config.capture.camera_fps
        ).motion_confirm_low_frames
        + 5
    )

    dinamicas: list[ReplayOutcome] = []
    estaticas: list[ReplayOutcome] = []
    for firmante in firmantes:
        registry = _registry_sin(corpus, config, firmante)
        for sample in corpus.samples:
            if sample.signer_id != firmante:
                continue
            if sample.label in PHASE5_LABELS and sample.kind is SampleKind.DYNAMIC:
                dinamicas.append(
                    replay_sample(sample, config, registry, rest_after=reposo)
                )
            elif sample.label in PHASE2_LABELS and sample.kind is SampleKind.STATIC:
                estaticas.append(
                    replay_sample(sample, config, registry, rest_after=reposo)
                )
    return tuple(dinamicas), tuple(estaticas)


def replay_static_without_dynamic(
    corpus: Corpus, config: Config
) -> tuple[ReplayOutcome, ...]:
    """Las estáticas por la máquina sin camino dinámico: la referencia v2."""
    referencia = without_dynamic_path(config)
    reposo = (
        FrameThresholds.from_config(
            config, config.capture.camera_fps
        ).motion_confirm_low_frames
        + 5
    )
    salida: list[ReplayOutcome] = []
    for firmante in sorted({s.signer_id for s in corpus.samples}):
        registry = _registry_sin(corpus, config, firmante)
        for sample in corpus.samples:
            if (
                sample.signer_id == firmante
                and sample.label in PHASE2_LABELS
                and sample.kind is SampleKind.STATIC
            ):
                salida.append(
                    replay_sample(sample, referencia, registry, rest_after=reposo)
                )
    return tuple(salida)


def _registry_sin(corpus: Corpus, config: Config, firmante: str) -> ClassifierRegistry:
    entrenamiento = [s for s in corpus.samples if s.signer_id != firmante]
    estatico = StaticKnnClassifier(config=config)
    estatico.fit([s for s in entrenamiento if s.label in set(PHASE2_LABELS)])
    dinamico = DynamicDtwClassifier(config=config)
    dinamico.fit(
        [
            s
            for s in entrenamiento
            if s.label in set(PHASE5_LABELS) and s.kind is SampleKind.DYNAMIC
        ]
    )
    return ClassifierRegistry(static=estatico, dynamic=dinamico)


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #


def _seccion_resultado(report: Report, labels: Sequence[str]) -> str:
    filas = [
        [
            label,
            str(score.total),
            f"{score.accuracy:.4f}",
            f"{score.unknown / score.total:.4f}" if score.total else "—",
        ]
        for label, score in report.per_label.items()
    ]
    columnas = [*labels, UNKNOWN_LABEL]
    matriz = [
        [verdad, *(str(report.confusion.get((verdad, p), 0)) for p in columnas)]
        for verdad in labels
    ]
    return "\n\n".join(
        [
            "## 1. Resultado con la configuración actual",
            (
                f"accuracy **{report.accuracy:.4f}** · macro "
                f"{report.macro_accuracy:.4f} · UNKNOWN {report.unknown_rate:.4f} "
                f"· {report.total} muestras"
            ),
            _tabla(["letra", "muestras", "accuracy", "UNKNOWN"], filas),
            "Matriz de confusión (fila: verdad, columna: predicha):",
            _tabla(["", *columnas], matriz),
        ]
    )


def _seccion_barrido(points: Sequence[DynamicGridPoint], labels: Sequence[str]) -> str:
    pesos = sorted({p.trajectory_weight for p in points})
    bandas = sorted({p.band_radius for p in points})
    por_clave = {(p.trajectory_weight, p.band_radius): p for p in points}

    def rejilla(valor: Callable[[DynamicGridPoint], str]) -> str:
        filas = [
            [f"w_τ = {peso:g}", *(valor(por_clave[(peso, b)]) for b in bandas)]
            for peso in pesos
        ]
        return _tabla(["", *(f"banda {b}" for b in bandas)], filas)

    def acc_de(label: str) -> Callable[[DynamicGridPoint], str]:
        def valor(p: DynamicGridPoint) -> str:
            score = p.report.per_label.get(label)
            return f"{score.accuracy:.3f}" if score else "—"

        return valor

    partes = [
        "## 2. Barrido `trajectory_weight × band_radius` (puerta abierta)",
        (
            "Accuracy del vecino más cercano, leave-one-signer-out, sin "
            "`max_distance`: la puerta depende de `w_τ` y compararía otra cosa."
        ),
        "### Todas las letras",
        rejilla(lambda p: f"{p.report.accuracy:.3f}"),
        "### Z, aparte (glosario, PENDIENTE-HUMANO F)",
        rejilla(acc_de("Z")),
    ]
    for label in labels:
        if label == "Z":
            continue
        partes += [f"### {label}", rejilla(acc_de(label))]
    return "\n\n".join(partes)


def _seccion_puerta(
    correctos: tuple[float, ...], negativos: tuple[float, ...], gates: Sequence[Gate]
) -> str:
    def q(valores: tuple[float, ...]) -> str:
        return " · ".join(
            f"p{int(p * 100)} {_percentil(valores, p):.3f}"
            for p in (0.05, 0.5, 0.9, 0.95, 0.99)
        )

    filas = [
        [
            f"{g.threshold:.3f}",
            g.origin,
            f"{g.positives_kept:.4f}",
            f"{g.negatives_accepted:.4f}",
        ]
        for g in gates
    ]
    return "\n\n".join(
        [
            "## 3. `dtw.max_distance`",
            f"d₁ de los aciertos (LOSO): {q(correctos)}",
            (
                f"d₁ de las {len(negativos)} muestras estáticas contra las "
                f"plantillas de producción: {q(negativos)}"
            ),
            _tabla(
                ["umbral", "origen", "aciertos que pasan", "estáticas que pasan"],
                filas,
            ),
        ]
    )


def _resumen_replay(salidas: Sequence[ReplayOutcome]) -> dict[str, dict[str, int]]:
    """Por letra: enteras, partidas, perdidas, y emitidas correctamente."""
    resumen: dict[str, dict[str, int]] = {}
    for r in salidas:
        fila = resumen.setdefault(
            r.label,
            {"muestras": 0, "enteras": 0, "partidas": 0, "perdidas": 0, "acierto": 0},
        )
        fila["muestras"] += 1
        if r.dynamic_windows == 1:
            fila["enteras"] += 1
        elif r.dynamic_windows > 1:
            fila["partidas"] += 1
        else:
            fila["perdidas"] += 1
        if [label for label, _ in r.emitted] == [r.label]:
            fila["acierto"] += 1
    return dict(sorted(resumen.items()))


def _seccion_replay(
    dinamicas: Sequence[ReplayOutcome],
    estaticas: Sequence[ReplayOutcome],
    referencia: Sequence[ReplayOutcome],
) -> str:
    resumen = _resumen_replay(dinamicas)
    filas = [
        [
            label,
            *(
                str(f[k])
                for k in ("muestras", "enteras", "partidas", "perdidas", "acierto")
            ),
        ]
        for label, f in resumen.items()
    ]

    def primera(r: ReplayOutcome) -> str | None:
        return r.emitted[0][0] if r.emitted else None

    cambiaron = sum(
        1 for a, b in zip(estaticas, referencia, strict=True) if a.emitted != b.emitted
    )
    candidatas = sum(1 for r in estaticas if r.dynamic_windows > 0)
    por_camino_dinamico = sum(
        1 for r in estaticas if any(o is WindowOrigin.DYNAMIC for _, o in r.emitted)
    )
    acierto_v3 = sum(1 for r in estaticas if primera(r) == r.label)
    acierto_v2 = sum(1 for r in referencia if primera(r) == r.label)
    total = len(estaticas) or 1
    return "\n\n".join(
        [
            "## 4. Reproducción por la máquina de estados (LOSO)",
            (
                "Cada grabación se pasa por `run_segmentation` con modelos "
                "entrenados sin su firmante, seguida de reposo. **Entera**: un "
                "solo trazo entregado. **Partida**: más de uno. **Perdida**: "
                "ninguno. **Acierto**: se emitió exactamente la letra grabada."
            ),
            _tabla(
                ["letra", "muestras", "enteras", "partidas", "perdidas", "acierto"],
                filas,
            ),
            "### Estáticas: con camino dinámico (v3) contra sin él (v2)",
            _tabla(
                ["", "valor"],
                [
                    ["muestras estáticas", str(len(estaticas))],
                    [
                        "entraron a candidato dinámico",
                        f"{candidatas} ({candidatas / total:.2%})",
                    ],
                    ["emitieron algo por el camino dinámico", str(por_camino_dinamico)],
                    [
                        "salida distinta a la v2",
                        f"{cambiaron} ({cambiaron / total:.2%})",
                    ],
                    ["primera letra correcta, v3", f"{acierto_v3 / total:.4f}"],
                    ["primera letra correcta, v2", f"{acierto_v2 / total:.4f}"],
                ],
            ),
        ]
    )


def _cabecera(corpus: Corpus, config: Config, protocol: Protocol) -> str:
    p = corpus.provenance
    lineas = [
        "# Reporte de la Fase 5 — letras dinámicas",
        "",
        f"- corpus: `{p.source}` · huella `{p.fingerprint}` · commit `{p.git_commit}`",
        f"- protocolo: {protocol.value}",
        (
            f"- configuración: w_τ = {config.features.trajectory_weight:g}, "
            f"band_radius = {config.dtw.band_radius}, "
            f"max_distance = {config.dtw.max_distance:g}"
        ),
        (
            f"- segmentación: motion_threshold_per_s = "
            f"{config.segmentation.motion_threshold_per_s:g}, motion_min_ms = "
            f"{config.segmentation.motion_min_ms:g}, motion_confirm_low_ms = "
            f"{config.segmentation.motion_confirm_low_ms:g}, motion_max_ms = "
            f"{config.segmentation.motion_max_ms:g}"
        ),
    ]
    if p.synthetic:
        lineas += ["", f"> **{SYNTHETIC_WARNING}**"]
    return "\n".join(lineas)


def _seccion_alcance(dataset: DynamicDataset) -> str:
    bloqueadas = ", ".join(f"{k} ({v})" for k, v in dataset.blocked.items()) or "—"
    return "\n\n".join(
        [
            "## Alcance",
            (
                f"Evaluadas: {', '.join(dataset.labels)} · "
                f"{len(dataset.observations)} muestras."
            ),
            (
                f"**Bloqueadas por dirección pendiente** (glosario, PENDIENTE-HUMANO "
                f"I): {bloqueadas}. Sin plantilla hasta que se decida su dirección "
                "canónica."
            ),
            f"Sin canal dinámico: {sum(dataset.rejected.values())}.",
        ]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-eval-dinamico",
        description=(
            "Evalúa el clasificador dinámico (DTW) con leave-one-signer-out, "
            "barre trajectory_weight × band_radius y reproduce las grabaciones "
            "por la máquina de estados. No necesita cámara."
        ),
    )
    parser.add_argument("--raiz", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protocolo",
        choices=[p.value for p in Protocol],
        default=Protocol.SIGNER.value,
    )
    parser.add_argument("--rejilla", choices=sorted(GRIDS), default="completo")
    parser.add_argument(
        "--sin-reproduccion",
        action="store_true",
        help="salta la sección 4, que es la más lenta",
    )
    parser.add_argument("--sin-sintetico", action="store_true")
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
            # Plausibilidad y huecos cortos, como en vivo, a la tasa nominal:
            # la de la grabación (ADR 0021, ADR 0027).
            preprocessing=Preprocessing.from_config(config, config.capture.camera_fps),
            exclude_before=config.corpus.exclude_before,
            dynamic_labels=PHASE5_LABELS,
        )
    except CorpusError as error:
        print(f"lsm-eval-dinamico: {error}")
        return 1

    dataset = observe_dynamic(corpus.samples, config)
    try:
        build_folds(dataset, protocol)
    except InsufficientFoldsError as error:
        print(f"lsm-eval-dinamico: {error}")
        return 1

    folds = precompute_dynamic(dataset, protocol, config.dtw.band_radius)
    report = score_dynamic(folds, config.dtw.max_distance)
    correctos = nearest_distances(folds, correct=True)

    estaticas = [
        s
        for s in corpus.samples
        if s.kind is SampleKind.STATIC and s.label in PHASE2_LABELS
    ]
    negativos, gates = measure_gate(dataset, correctos, estaticas, config)

    pesos, bandas = GRIDS[args.rejilla]
    points = sweep_dynamic(corpus.samples, config, pesos, bandas, protocol)

    secciones = [
        _cabecera(corpus, config, protocol),
        _seccion_alcance(dataset),
        _seccion_resultado(report, dataset.labels),
        _seccion_barrido(points, dataset.labels),
        _seccion_puerta(correctos, negativos, gates),
    ]
    replay: dict[str, Any] = {}
    if not args.sin_reproduccion:
        dinamicas, estaticas_v3 = replay_loso(corpus, config)
        estaticas_v2 = replay_static_without_dynamic(corpus, config)
        secciones.append(_seccion_replay(dinamicas, estaticas_v3, estaticas_v2))
        replay = {
            "dynamic": _resumen_replay(dinamicas),
            "static_changed": sum(
                1
                for a, b in zip(estaticas_v3, estaticas_v2, strict=True)
                if a.emitted != b.emitted
            ),
            "static_total": len(estaticas_v3),
        }

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "reporte-fase5.md").write_text(
        "\n\n".join(secciones) + "\n", encoding="utf-8"
    )
    resultados = {
        "dataset": corpus.provenance.to_json(),
        "protocol": protocol.value,
        "blocked_labels": sorted(DIRECTION_PENDING_LABELS),
        "config": {
            "trajectory_weight": config.features.trajectory_weight,
            "band_radius": config.dtw.band_radius,
            "max_distance": config.dtw.max_distance,
        },
        "accuracy": report.accuracy,
        "macro_accuracy": report.macro_accuracy,
        "unknown_rate": report.unknown_rate,
        "per_label": {
            k: {"total": v.total, "correct": v.correct, "unknown": v.unknown}
            for k, v in report.per_label.items()
        },
        "sweep": [
            {
                "trajectory_weight": p.trajectory_weight,
                "band_radius": p.band_radius,
                "accuracy": p.report.accuracy,
                "per_label": {k: v.accuracy for k, v in p.report.per_label.items()},
                "correct_nearest_p99": _percentil(p.correct_nearest, 0.99),
            }
            for p in points
        ],
        "gate": [
            {
                "threshold": g.threshold,
                "origin": g.origin,
                "positives_kept": g.positives_kept,
                "negatives_accepted": g.negatives_accepted,
            }
            for g in gates
        ],
        "replay": replay,
    }
    (args.salida / "resultados-fase5.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if corpus.provenance.synthetic:
        print(SYNTHETIC_WARNING)
    print(
        f"accuracy {report.accuracy:.4f} · macro {report.macro_accuracy:.4f} · "
        f"UNKNOWN {report.unknown_rate:.4f} · {report.total} muestras"
    )
    print(f"reporte en {args.salida / 'reporte-fase5.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
