"""Enrutamiento entre el clasificador estático y el dinámico (`ARQUITECTURA.md` §4.3).

`ARQUITECTURA.md` proponía dos formas de decidir: por la energía de movimiento de
la ventana, o corriendo ambos y quedándose con la mayor confianza. **Ninguna de
las dos se toma aquí.**

- **No se re-mide el movimiento.** Esa decisión ya la tomó la máquina de estados
  (`segmentation.py`, §6.7): una ventana que salió de STABLE es una forma
  sostenida y una que salió de DYNAMIC_EMIT es un trazo. Si el registry midiera
  la energía otra vez, con su propio umbral o su propia ventana, habría dos
  criterios para lo mismo, y el día que discreparan una J iría al estático sin
  que nada lo dijera. La ruta llega como `WindowOrigin` y se obedece.
- **No se corren ambos por defecto.** El glosario ya lo anotó: para `L`/`LL`,
  `R`/`RR`, `N`/`Ñ` e `I`/`J` la forma de la mano es la misma, así que el
  estático diría `L` con confianza alta ante una `LL`. «La mayor confianza»
  elegiría la letra equivocada con toda seguridad.

El caso de «ambas rutas para el mismo evento» existe solo como red —la máquina
de estados no lo produce, porque sus dos caminos son excluyentes—, y ahí sí se
corren los dos y gana la mayor confianza, como decía el diseño original
(`classify_routes`).

Un origen sin clasificador cargado devuelve UNKNOWN. Es lo que pasa en la demo
si solo hay modelo estático: los trazos se rechazan en vez de caer al
clasificador equivocado.

Código puro: los modelos llegan ya cargados. Leer los JSON es de `lsm.cli`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lsm.classifiers import dynamic_dtw, static_knn
from lsm.classifiers.base import Classifier
from lsm.classifiers.dynamic_dtw import DynamicDtwClassifier
from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.types import Prediction, Sequence, WindowOrigin

#: Qué implementación atiende cada origen. Es el único sitio donde se escribe
#: esa correspondencia: `from_exports` la usa para rechazar un modelo cargado en
#: la ranura equivocada.
EXPECTED_CLASSIFIER: dict[WindowOrigin, str] = {
    WindowOrigin.STABLE: static_knn.CLASSIFIER_NAME,
    WindowOrigin.DYNAMIC: dynamic_dtw.CLASSIFIER_NAME,
}


@dataclass(frozen=True)
class ClassifierRegistry:
    """Un clasificador por origen de ventana. Se pasa tal cual como `Classify`."""

    static: Classifier | None = None
    dynamic: Classifier | None = None

    def classifier_for(self, origin: WindowOrigin) -> Classifier | None:
        return self.static if origin is WindowOrigin.STABLE else self.dynamic

    def __call__(self, sequence: Sequence, origin: WindowOrigin) -> Prediction:
        """La firma de `segmentation.Classify`: una ventana, un origen."""
        return self.classify_routes(sequence, (origin,))

    def classify_routes(
        self, sequence: Sequence, origins: Iterable[WindowOrigin]
    ) -> Prediction:
        """Corre el clasificador de cada origen y se queda con la mayor confianza.

        Con un solo origen es el enrutamiento normal. Con los dos es la red del
        caso que la máquina de estados no debería producir. Solo compiten las
        predicciones que no son UNKNOWN; si todas lo son, sale UNKNOWN con la
        mayor confianza que se vio, para que el HUD siga mostrando algo.

        Desempate en el orden de `WindowOrigin` —estático primero—: dos
        clasificadores exactamente igual de seguros son raros, pero el resultado
        tiene que ser el mismo en dos ejecuciones.
        """
        rutas = [o for o in WindowOrigin if o in set(origins)]
        if not rutas:
            raise ValueError("hace falta al menos un origen que enrutar")

        mejor: Prediction | None = None
        rechazo = Prediction.unknown()
        for origin in rutas:
            classifier = self.classifier_for(origin)
            if classifier is None:
                continue
            prediction = classifier.predict(sequence)
            if prediction.is_unknown:
                if prediction.confidence > rechazo.confidence:
                    rechazo = prediction
                continue
            if mejor is None or prediction.confidence > mejor.confidence:
                mejor = prediction
        return mejor if mejor is not None else rechazo

    @classmethod
    def from_exports(
        cls,
        static: Mapping[str, Any] | None = None,
        dynamic: Mapping[str, Any] | None = None,
    ) -> ClassifierRegistry:
        """Construye el registry desde los JSON exportados, cada uno en su ranura.

        Un modelo dinámico en la ranura estática —o al revés— se rechaza aquí y
        no al predecir: `from_export` de cada clase ya verifica su propio nombre,
        pero el mensaje de este nivel dice cuál de los dos archivos está mal.
        """
        for origin, payload in (
            (WindowOrigin.STABLE, static),
            (WindowOrigin.DYNAMIC, dynamic),
        ):
            if (
                payload is not None
                and payload.get("classifier") != (EXPECTED_CLASSIFIER[origin])
            ):
                msg = (
                    f"la ranura {origin.value} espera un modelo "
                    f"{EXPECTED_CLASSIFIER[origin]!r} y recibió "
                    f"{payload.get('classifier')!r}"
                )
                raise ValueError(msg)
        return cls(
            static=StaticKnnClassifier.from_export(static) if static else None,
            dynamic=DynamicDtwClassifier.from_export(dynamic) if dynamic else None,
        )
