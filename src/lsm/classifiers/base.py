"""Contrato común de los clasificadores (`ARQUITECTURA.md` §4.3 y §4.6).

Habrá al menos dos implementaciones —`static_knn` para las letras sin movimiento
y `dynamic_dtw` para las que trazan un recorrido— y probablemente una tercera si
la matriz de confusión lo pide. Todas entran por esta puerta, de modo que
cambiarlas no toque ni la captura, ni el entrenamiento, ni la demo.

En Fase 0 solo existe el contrato y un doble de pruebas. Lo que sí queda cerrado
es el formato de export, porque de él dependen dos cosas que salen caras si se
descubren tarde: que el modelo entrenado en Python corra igual en el navegador, y
que un modelo entrenado con otras premisas se **rechace** al cargarse en vez de
devolver predicciones malas sin ningún síntoma.

Hay dos premisas que se comprueban así, y las dos fallan en silencio:

- `feature_spec_version` — si cambia la normalización, los números del modelo
  dejan de significar lo que significaban.
- `detector_input` — que el cuadro llega al detector **sin espejar**
  (`feature-spec.md` §0.3). Desde `FEATURE_SPEC_VERSION` 2 el espejo del paso 2
  lo decide la mano declarada, no la etiqueta del detector, así que lo que dos
  implementaciones tienen que compartir ya no es qué nombre le pone MediaPipe a
  la mano sino cómo le llega la imagen: con un cuadro espejado, una derecha
  declarada se vería como izquierda y el modelo confundiría cada seña con su
  espejo. **Los golden vectors no lo detectan**: reciben landmarks ya
  detectados. Reemplaza a la antigua `handedness_convention` (ADR 0017).

Código puro: sin disco, sin cámara, sin MediaPipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Protocol, runtime_checkable

from lsm.features import FEATURE_SPEC_VERSION
from lsm.types import DETECTOR_INPUT, Prediction, Sample, Sequence

#: Versión del formato de archivo del modelo exportado. Cambia cuando cambia la
#: estructura del JSON; es independiente de `feature_spec_version`, que cambia
#: cuando cambia el significado de los números que van dentro.
SCHEMA_VERSION: Final = 1

#: Campos obligatorios del export (`ARQUITECTURA.md` §4.6).
REQUIRED_EXPORT_FIELDS: Final = (
    "schema_version",
    "feature_spec_version",
    "detector_input",
    "classifier",
    "labels",
    "params",
    "data",
)


class IncompatibleModelError(RuntimeError):
    """El modelo no se puede ejecutar con este runtime.

    Se lanza al **cargar**, nunca al predecir: un modelo entrenado con otra
    normalización no falla ruidosamente, simplemente acierta menos, y eso es
    exactamente lo que no se detecta en una demo.
    """


@runtime_checkable
class Classifier(Protocol):
    """Interfaz que implementa todo clasificador del proyecto."""

    def fit(self, samples: list[Sample]) -> None:
        """Entrena con muestras etiquetadas.

        Recibe `Sample`, no vectores de features: el clasificador decide qué
        extrae de la secuencia, y los metadatos que lleva la muestra son los que
        permiten construir el split leave-one-signer-out.
        """
        ...

    def predict(self, sequence: Sequence) -> Prediction:
        """Clasifica una secuencia `(T, 21, 3)`.

        Nunca recibe un frame suelto. Devuelve siempre una `Prediction` con
        confianza, y puede devolver `UNKNOWN`: un clasificador de 27 clases que
        no sabe decir "esto no es una letra" escribe basura cada vez que alguien
        se rasca la nariz.
        """
        ...

    def export(self) -> dict[str, Any]:
        """Serializa el modelo a un dict JSON-serializable.

        Lo va a leer JavaScript: nada de tuplas, enums, `datetime` ni arreglos de
        numpy. Debe llevar los campos de `REQUIRED_EXPORT_FIELDS`.
        """
        ...


def build_export(
    *,
    classifier: str,
    labels: list[str],
    params: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Arma el dict de export con las dos versiones ya puestas.

    Existe para que ninguna implementación se invente el formato ni olvide
    `feature_spec_version`, que es el campo del que depende todo el mecanismo de
    rechazo.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "detector_input": DETECTOR_INPUT,
        "classifier": classifier,
        "labels": labels,
        "params": params,
        "data": data,
    }


def check_export_compatibility(payload: Mapping[str, Any]) -> None:
    """Verifica que un modelo exportado se pueda ejecutar con este runtime.

    Lanza `IncompatibleModelError` si falta un campo o si alguna de las dos
    versiones no coincide. No hay migración automática y es deliberado: reentrenar
    cuesta minutos, depurar un modelo que corre con la normalización equivocada
    cuesta días.
    """
    for field in REQUIRED_EXPORT_FIELDS:
        if field not in payload:
            msg = f"el modelo exportado no trae el campo obligatorio '{field}'"
            raise IncompatibleModelError(msg)

    if payload["schema_version"] != SCHEMA_VERSION:
        msg = (
            f"schema_version {payload['schema_version']} incompatible: "
            f"este runtime lee {SCHEMA_VERSION}"
        )
        raise IncompatibleModelError(msg)

    if payload["feature_spec_version"] != FEATURE_SPEC_VERSION:
        msg = (
            f"feature_spec_version {payload['feature_spec_version']} incompatible: "
            f"este runtime extrae features con la versión {FEATURE_SPEC_VERSION}. "
            "Reentrenar el modelo; ejecutarlo así daría predicciones malas en "
            "silencio."
        )
        raise IncompatibleModelError(msg)

    if payload["detector_input"] != DETECTOR_INPUT:
        msg = (
            f"detector_input {payload['detector_input']!r} incompatible: este "
            f"runtime alimenta al detector con cuadros {DETECTOR_INPUT!r}. Un "
            "modelo entrenado con la imagen espejada confundiría cada seña con su "
            "espejo, y con la misma confianza que si acertara."
        )
        raise IncompatibleModelError(msg)
