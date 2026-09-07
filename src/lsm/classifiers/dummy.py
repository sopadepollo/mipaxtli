"""Doble de pruebas del `Protocol` de clasificador.

No aprende nada y no pretende hacerlo: existe para que la máquina de estados y el
formato de export se puedan testear en Fase 0, antes de que haya un dataset. Las
implementaciones reales —`static_knn` y `dynamic_dtw`— llegan en las Fases 2 y 5.

Devuelve predicciones guionadas, lo que permite escribir tests donde el
clasificador acierta, duda o se equivoca a voluntad, sin depender de datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lsm.classifiers.base import build_export
from lsm.types import Prediction, Sample, Sequence


@dataclass
class DummyClassifier:
    """Clasificador guionado. Cumple el contrato, no resuelve el problema."""

    #: Predicciones a devolver, en orden. La última se repite indefinidamente.
    #: Vacío significa "siempre UNKNOWN".
    responses: tuple[Prediction, ...] = ()

    _labels: list[str] = field(default_factory=list, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)

    def fit(self, samples: list[Sample]) -> None:
        """Se queda solo con el vocabulario de etiquetas del dataset."""
        self._labels = sorted({sample.label for sample in samples})

    def predict(self, sequence: Sequence) -> Prediction:  # noqa: ARG002 — guionado
        if not self.responses:
            return Prediction.unknown()
        index = min(self._calls, len(self.responses) - 1)
        self._calls += 1
        return self.responses[index]

    def export(self) -> dict[str, Any]:
        return build_export(
            classifier="dummy",
            labels=list(self._labels),
            params={"responses": len(self.responses)},
            data={},
        )
