"""`make train` — entrena `static_knn` y `dynamic_dtw` y escribe los dos modelos.

Este comando **no reporta precisión**, a propósito. Un accuracy sobre las mismas
muestras con las que se entrenó es siempre estupendo y no significa nada; el
número honesto lo da `make eval` con validación leave-one-signer-out. Aquí solo se
cuentan muestras, se avisa de lo que se descartó y se escribe el archivo.

Lo que sí es responsabilidad suya es que el artefacto quede **trazable**: el JSON
lleva dentro la huella del corpus con el que se entrenó. Sin eso, un modelo en
`data/models/` y unas grabaciones en `data/raw/` son dos cosas que uno *supone*
que se corresponden.

Dos modelos, dos archivos, dos alcances:

- `static_knn.json`: las 21 letras estáticas más `NONE`
  (`lsm.evaluation.PHASE2_LABELS`).
- `dynamic_dtw.json`: las dinámicas con dirección canónica decidida
  (`lsm.evaluation.PHASE5_LABELS`). `LL` y `RR` se cuentan como **bloqueadas**
  y no tienen plantilla: su dirección está pendiente de decisión humana
  (`docs/glosario-lsm.md`). La demo carga los dos a través del registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lsm.classifiers.dynamic_dtw import DynamicDtwClassifier
from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.config import load_config
from lsm.evaluation import (
    EXCLUDED_LABELS,
    PHASE2_LABELS,
    PHASE5_LABELS,
    Dataset,
    observe,
)
from lsm.io.corpus import (
    SYNTHETIC_REPETITIONS,
    SYNTHETIC_SIGNERS,
    SYNTHETIC_WARNING,
    Corpus,
    CorpusError,
    Provenance,
    load_corpus,
)
from lsm.types import SampleKind

DEFAULT_MODEL = Path("data/models/static_knn.json")
DEFAULT_DYNAMIC_MODEL = Path("data/models/dynamic_dtw.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-train",
        description=(
            "Entrena el clasificador de letras estáticas y exporta el modelo a "
            "JSON. No necesita cámara ni MediaPipe."
        ),
    )
    parser.add_argument(
        "--raiz",
        type=Path,
        default=Path("data/raw"),
        help="raíz del dataset (por defecto: data/raw)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="ruta de config.yaml"
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=DEFAULT_MODEL,
        help="dónde escribir el modelo estático",
    )
    parser.add_argument(
        "--salida-dinamico",
        type=Path,
        default=DEFAULT_DYNAMIC_MODEL,
        dest="salida_dinamico",
        help="dónde escribir el modelo dinámico",
    )
    parser.add_argument(
        "--sin-sintetico",
        action="store_true",
        help=(
            "falla si data/raw está vacío en vez de generar el corpus sintético "
            "de prueba"
        ),
    )
    parser.add_argument(
        "--repeticiones-sinteticas",
        type=int,
        default=SYNTHETIC_REPETITIONS,
        help="repeticiones por letra del corpus sintético",
    )
    parser.add_argument(
        "--firmantes-sinteticos",
        type=int,
        default=SYNTHETIC_SIGNERS,
        help="firmantes simulados del corpus sintético",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    try:
        corpus = load_corpus(
            args.raiz,
            PHASE2_LABELS,
            allow_synthetic=not args.sin_sintetico,
            repetitions=args.repeticiones_sinteticas,
            signers=args.firmantes_sinteticos,
            dynamic_labels=PHASE5_LABELS,
        )
    except CorpusError as error:
        print(f"lsm-train: {error}")
        return 1

    if corpus.provenance.synthetic:
        print(SYNTHETIC_WARNING)
        print()

    dataset = observe(corpus.samples, config)
    if not dataset.observations:
        print(
            "lsm-train: ninguna muestra sobrevivió al filtro de alcance y a la "
            "extracción de features; no hay nada que entrenar."
        )
        return 1

    classifier = StaticKnnClassifier(config=config)
    classifier.fit(
        [sample for sample in corpus.samples if sample.label in set(PHASE2_LABELS)]
    )

    _escribir(classifier.export(), corpus.provenance, args.salida)

    # Todas las dinámicas, también las bloqueadas: el clasificador es quien
    # aplica el bloqueo y las cuenta, y así el resumen puede decir cuántas
    # muestras de LL y RR esperan una decisión.
    dinamico = DynamicDtwClassifier(config=config)
    dinamico.fit(
        [
            s
            for s in corpus.samples
            if s.label in EXCLUDED_LABELS and s.kind is SampleKind.DYNAMIC
        ]
    )
    _escribir(dinamico.export(), corpus.provenance, args.salida_dinamico)

    _resumen(corpus, dataset, classifier, args.salida)
    _resumen_dinamico(dinamico, args.salida_dinamico)
    return 0


def _escribir(payload: dict[str, Any], provenance: Provenance, salida: Path) -> None:
    """El JSON del modelo con la huella del corpus dentro."""
    payload["data"]["dataset"] = provenance.to_json()
    if provenance.synthetic:
        payload["data"]["dataset"]["warning"] = SYNTHETIC_WARNING
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resumen_dinamico(dinamico: DynamicDtwClassifier, salida: Path) -> None:
    etiquetas = sorted({t.label for t in dinamico.templates})
    print()
    print(
        f"dinámico          {len(dinamico.templates)} plantillas de "
        f"{', '.join(etiquetas) or '—'}"
    )
    if dinamico.blocked:
        detalle = ", ".join(f"{k}={v}" for k, v in dinamico.blocked.items())
        print(
            f"  bloqueadas:     {detalle}  <- dirección canónica pendiente "
            "(docs/glosario-lsm.md, PENDIENTE-HUMANO I)"
        )
    if dinamico.rejected:
        print(f"  sin canal:      {dinamico.rejected}")
    print(f"modelo dinámico   {salida}")


def _resumen(
    corpus: Corpus,
    dataset: Dataset,
    classifier: StaticKnnClassifier,
    salida: Path,
) -> None:
    print(f"corpus            {corpus.provenance.source}")
    print(f"huella            {corpus.provenance.fingerprint}")
    print(f"muestras          {corpus.provenance.sample_count}")
    print(f"firmantes         {', '.join(corpus.provenance.signers)}")
    print(
        f"clases entrenadas {len(classifier.export()['labels'])} de "
        f"{len(PHASE2_LABELS)}"
    )

    faltantes = set(PHASE2_LABELS) - set(classifier.export()["labels"])
    if faltantes:
        print(
            "  sin muestras:   "
            + ", ".join(sorted(faltantes))
            + "  <- el modelo no puede reconocerlas"
        )

    if dataset.excluded:
        total = sum(dataset.excluded.values())
        print(
            f"descartadas       {total} muestras de letras dinámicas "
            f"(Fase 5): {', '.join(sorted(dataset.excluded))}"
        )
    else:
        print(
            "descartadas       0 (el corpus no tiene ninguna de las "
            f"{len(EXCLUDED_LABELS)} letras dinámicas)"
        )

    if dataset.rejected:
        detalle = ", ".join(f"{k}={v}" for k, v in sorted(dataset.rejected.items()))
        print(f"rechazadas        {sum(dataset.rejected.values())} ({detalle})")

    print(f"modelo            {salida}")
    print()
    print(
        "La precisión honesta la dan `make eval` y `make eval-dinamico`: aquí no "
        "se mide nada."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
