"""De dónde salen las muestras que se entrenan y se evalúan, y cómo se identifica.

Dos consumidores —`lsm-train` y `lsm-eval`— tienen que estar de acuerdo en tres
cosas: qué muestras leen, cómo se llama ese conjunto de muestras, y qué hacen
cuando no hay ninguna. Si cada uno lo resolviera por su cuenta, un modelo podría
quedar entrenado con un corpus y evaluado con otro sin que nada lo dijera.

## La huella

`Provenance.fingerprint` es un SHA-256 del **contenido**, no de las rutas ni de la
fecha. Es lo que hace verificable la frase "estos umbrales se calibraron contra
este dataset": dos corpus con la misma huella son el mismo corpus, y un archivo de
calibración cuya huella no coincida con el dataset actual está describiendo otra
cosa.

Deliberadamente **no lleva marca de tiempo**. El criterio de aceptación de la
Fase 2 es que `make eval` sea reproducible, y un `now()` metido en el reporte lo
rompería en el primer segundo.

## El corpus sintético

Mientras `data/raw` esté vacío —la Fase 1 no ha cerrado— se genera un corpus
determinista con `lsm.synthetic`. **No se parece a LSM.** Existe para que el
barrido, la matriz de confusión y el contraste de hipótesis se puedan ejecutar,
depurar y testear hoy, en vez de escribirse a ciegas y estrenarse el día que
existan grabaciones. Todo lo que sale de él va marcado, en la huella y en la
cabecera del reporte.

Este módulo toca disco y llama a `git`, pero no importa OpenCV ni MediaPipe.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lsm.capture import CAPTURE_SPEC_VERSION
from lsm.features import FEATURE_SPEC_VERSION
from lsm.io.dataset import iter_sample_paths, read_sample
from lsm.segmentation import SEGMENTATION_SPEC_VERSION
from lsm.synthetic import synthetic_samples
from lsm.types import Sample

#: Cómo se nombra el corpus sintético en la huella y en el reporte.
SYNTHETIC_SOURCE: Final = "sintetico"

#: Cuántas repeticiones por letra, firmante y sesión genera el corpus sintético
#: por defecto. `ARQUITECTURA.md` §4.7 pide ~20 repeticiones reales; aquí bastan
#: menos, porque lo que se ejercita es la tubería y no la estadística.
SYNTHETIC_REPETITIONS: Final = 4

#: Firmantes y sesiones simulados. Tres firmantes es el mínimo que hace que
#: leave-one-signer-out tenga tres folds y no dos.
SYNTHETIC_SIGNERS: Final = 3
SYNTHETIC_SESSIONS: Final = 2


class CorpusError(RuntimeError):
    """No hay muestras con las que trabajar."""


@dataclass(frozen=True, slots=True)
class Provenance:
    """Contra qué se entrenó o se calibró. Va dentro de todo lo que se escribe."""

    source: str
    synthetic: bool
    #: SHA-256 del contenido del corpus. La identidad verificable del dataset.
    fingerprint: str
    sample_count: int
    signers: tuple[str, ...]
    sessions: tuple[str, ...]
    labels: tuple[str, ...]
    #: Commit del repositorio, si `git` está disponible. `None` dentro de un
    #: contenedor sin `.git`, y no es un error: la huella del contenido ya
    #: identifica los datos, esto identifica el código que los leyó.
    git_commit: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "synthetic": self.synthetic,
            "fingerprint": self.fingerprint,
            "sample_count": self.sample_count,
            "signers": list(self.signers),
            "sessions": list(self.sessions),
            "labels": list(self.labels),
            "git_commit": self.git_commit,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "segmentation_spec_version": SEGMENTATION_SPEC_VERSION,
            "capture_spec_version": CAPTURE_SPEC_VERSION,
        }

    @property
    def short(self) -> str:
        """Los doce primeros caracteres de la huella, para citarla en prosa."""
        return self.fingerprint[:12]


@dataclass(frozen=True, slots=True)
class Corpus:
    samples: tuple[Sample, ...]
    provenance: Provenance


def git_commit(repo: Path) -> str | None:
    """El commit actual, o `None` si no hay repositorio ni `git`."""
    try:
        salida = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if salida.returncode != 0:
        return None
    return salida.stdout.strip() or None


def _fingerprint_of_files(paths: list[Path], root: Path) -> str:
    """Huella de un dataset en disco: ruta relativa y hash de cada archivo.

    Incluye la ruta relativa además del contenido porque mover una muestra de un
    firmante a otro cambia el split sin cambiar un solo byte de landmarks, y ese
    corpus no es el mismo aunque sus archivos lo sean.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_of_samples(samples: tuple[Sample, ...]) -> str:
    """Huella de un corpus en memoria: sus etiquetas, su procedencia y sus números.

    Se recorren los landmarks de verdad y no los parámetros del generador. Si
    alguien cambia la geometría de `lsm.synthetic` sin acordarse de tocar una
    constante de versión, la huella cambia igualmente y el archivo de calibración
    deja de coincidir — que es exactamente lo que debe pasar.
    """
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            f"{sample.label}|{sample.signer_id}|{sample.session_id}\n".encode()
        )
        for frame in sample.sequence.frames:
            for landmark in frame.landmarks:
                digest.update(
                    f"{landmark.x!r},{landmark.y!r},{landmark.z!r};".encode("ascii")
                )
        digest.update(b"\n")
    return digest.hexdigest()


def _describe(
    samples: tuple[Sample, ...],
    *,
    source: str,
    synthetic: bool,
    fingerprint: str,
    repo: Path,
) -> Provenance:
    return Provenance(
        source=source,
        synthetic=synthetic,
        fingerprint=fingerprint,
        sample_count=len(samples),
        signers=tuple(sorted({s.signer_id for s in samples})),
        sessions=tuple(sorted({s.session_id for s in samples})),
        labels=tuple(sorted({s.label for s in samples})),
        git_commit=git_commit(repo),
    )


def load_corpus(
    root: Path,
    labels: tuple[str, ...],
    *,
    allow_synthetic: bool = True,
    repetitions: int = SYNTHETIC_REPETITIONS,
    signers: int = SYNTHETIC_SIGNERS,
    repo: Path | None = None,
) -> Corpus:
    """Lee `data/raw`; si está vacío y se permite, genera el corpus sintético.

    Levanta `CorpusError` cuando no hay muestras y el sintético está desactivado.
    El mensaje dice qué falta y cómo grabarlo: quien llegue aquí probablemente
    acaba de clonar el repositorio.
    """
    repo = repo or Path.cwd()
    paths = list(iter_sample_paths(root))

    if paths:
        samples = tuple(read_sample(path).to_sample() for path in paths)
        return Corpus(
            samples=samples,
            provenance=_describe(
                samples,
                source=root.as_posix(),
                synthetic=False,
                fingerprint=_fingerprint_of_files(paths, root),
                repo=repo,
            ),
        )

    if not allow_synthetic:
        msg = (
            f"no hay ninguna muestra en {root}. La Fase 1 graba el dataset: "
            "`make setup-capture`, `make model`, `make calibrar` y luego "
            "`make capture ARGS=...`. Para ejercitar la tubería sin grabar, "
            "quita --sin-sintetico."
        )
        raise CorpusError(msg)

    samples = synthetic_samples(
        labels,
        signers=signers,
        sessions=SYNTHETIC_SESSIONS,
        repetitions=repetitions,
    )
    return Corpus(
        samples=samples,
        provenance=_describe(
            samples,
            source=SYNTHETIC_SOURCE,
            synthetic=True,
            fingerprint=_fingerprint_of_samples(samples),
            repo=repo,
        ),
    )


#: Aviso que encabeza todo lo que se derive del corpus sintético. Se repite en el
#: reporte y en el modelo exportado a propósito: los dos archivos circulan por
#: separado y cualquiera de los dos puede acabar citado sin el otro al lado.
SYNTHETIC_WARNING: Final = (
    "CORPUS SINTÉTICO — estas cifras NO dicen nada sobre LSM. Las manos las "
    "generó `lsm.synthetic`, no una persona firmando. Sirven para comprobar que "
    "la tubería corre y que el reporte es reproducible; el número que importa "
    "sale cuando `data/raw` tenga grabaciones reales."
)
