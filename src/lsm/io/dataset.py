"""Lectura y escritura de muestras en `data/raw/`.

El formato y la disposición en disco están especificados en
`docs/dataset-schema.md`; este módulo es su implementación. Si los dos dejan de
coincidir, manda el documento.

**Se guardan landmarks crudos, nunca features.** Es la regla que hace que un
cambio de normalización cueste un comando y no una nueva ronda de grabaciones con
personas frente a la cámara (`ARQUITECTURA.md` §4.7). El coste es despreciable:
21 landmarks × 3 floats × T frames son unos pocos kilobytes por muestra.

**Y se guardan los huecos como huecos.** Un frame donde el detector no encontró la
mano se escribe con su centinela, no se omite: si se perdiera al guardar, una
secuencia interrumpida se convertiría en una continua y el archivo mentiría sobre
lo que ocurrió frente a la cámara.

Este módulo toca disco pero no importa OpenCV ni MediaPipe: es JSON y rutas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from lsm.capture import CAPTURE_SPEC_VERSION
from lsm.features import split_valid_runs
from lsm.io.hands import frames_from_json, frames_to_json
from lsm.preprocessing import NotReconstructed, Preprocessing, reconstruct
from lsm.segmentation import SEGMENTATION_SPEC_VERSION
from lsm.types import (
    HANDEDNESS_CONVENTION,
    Distance,
    FrameStream,
    Handedness,
    HandednessConvention,
    LightDirection,
    LightLevel,
    Sample,
    SampleKind,
    Sequence,
)

#: Versión del formato de archivo de una muestra. Un archivo de otra versión se
#: rechaza al cargar, igual que un modelo exportado con otro `feature_spec`: leer
#: un esquema viejo "lo mejor que se pueda" es la forma de meter basura en el
#: dataset sin que nada avise.
#:
#: **v3** (ADR 0017): `handedness` es la mano **declarada** por quien firma, y
#: `handedness_source` dice de dónde salió. La v2 se sigue leyendo —es todo el
#: dataset de la Fase 1—, con `handedness_source = DETECTED`: ahí `handedness`
#: es la etiqueta de MediaPipe, que la captura exigía constante en toda la
#: muestra.
#:
#: **v4** (Bloque 4, ADR 0023): `stroke_frames`. Una dinámica grabada desde el
#: Bloque 4 guarda el trazo que entregó la máquina de estados **y el reposo que
#: lo cerró**; `stroke_frames` dice cuántos frames del principio son el trazo, que
#: es lo que se entrena. `null` en las estáticas y en las dinámicas anteriores.
#:
#: **v5** (ADR 0018): cada entrada de `frames`, válida o hueco, lleva `"t_ms"`,
#: la marca de tiempo que puso el detector. Las anteriores se leen sin marcas y
#: quien necesita el tiempo usa el del índice a la tasa nominal: no se inventan
#: marcas que no se midieron.
SAMPLE_SCHEMA_VERSION: Final = 5

#: Versiones que se leen. Solo la actual se escribe.
READABLE_SAMPLE_SCHEMAS: Final = frozenset({2, 3, 4, 5})

#: Manifiesto de las dinámicas grabadas antes del Bloque 4 que terminan con la
#: mano todavía en movimiento (`lsm-capture marcar-truncadas`). Vive en la raíz
#: del dataset, fuera de `<firmante>/<sesion>/<letra>/`, así que no es una
#: muestra. Las listadas no se borran: se excluyen del entrenamiento y la
#: evaluación al cargar el corpus.
TRUNCATED_MANIFEST: Final = "truncadas.json"
TRUNCATED_MANIFEST_VERSION: Final = 1


class HandSource(StrEnum):
    """De dónde sale la `handedness` de una muestra."""

    #: La declaró quien firma al abrir la sesión (`--mano`). Schema v3.
    DECLARED = "DECLARED"
    #: La dijo MediaPipe, igual en todos los frames (la captura rechazaba las
    #: muestras con lateralidad mixta). Todo el dataset grabado antes del ADR
    #: 0017. Si MediaPipe se equivocó de forma sostenida en una muestra entera,
    #: aquí no hay manera de saberlo.
    DETECTED = "DETECTED"


#: Intentos dinámicos que la captura rechazó (Paso 0, ADR 0024). Viven en una
#: subcarpeta de la letra, `<firmante>/<sesion>/<LETRA>/rechazados/NNN.json`:
#: un nivel más abajo que las muestras, así que `iter_sample_paths` no los
#: alcanza y el entrenamiento no los ve. Además no son muestras: su archivo
#: tiene otro esquema y `read_sample` los rechaza.
REJECTED_DIRNAME: Final = "rechazados"
REJECTED_ATTEMPT_SCHEMA_VERSION: Final = 1

#: Versión del registro de consentimiento.
CONSENT_SCHEMA_VERSION: Final = 1

#: Nombre del registro de consentimiento, en la raíz de `data/raw/`.
CONSENT_FILENAME: Final = "consentimiento.json"

#: Subcarpeta de las sesiones de prueba. Vive dentro de la raíz para que
#: comparta el consentimiento y la calibración —son del mismo equipo y de las
#: mismas personas— pero un nivel más abajo, de modo que el recorrido del
#: dataset formal no la alcanza. Ver `iter_sample_paths`.
TRIAL_DIRNAME: Final = "pruebas"

#: `signer_id` y `session_id` viajan a nombres de carpeta y a claves JSON, así que
#: se restringen a lo que sobrevive a cualquier sistema de archivos. Además
#: impide que un `../` en un identificador escriba fuera de `data/raw/`.
_IDENTIFICADOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class DatasetError(RuntimeError):
    """El dataset en disco no es lo que este código sabe leer o escribir."""


def check_identifier(nombre: str, valor: str) -> str:
    """Valida un `signer_id` o `session_id` antes de que llegue a una ruta."""
    if not _IDENTIFICADOR.match(valor):
        msg = (
            f"{nombre} inválido: {valor!r}. Se admiten letras, dígitos, guion y "
            "guion bajo, empezando por letra o dígito, hasta 64 caracteres. "
            "Viaja a un nombre de carpeta."
        )
        raise DatasetError(msg)
    return valor


@dataclass(frozen=True, slots=True)
class SampleMetadata:
    """Los metadatos de `docs/dataset-schema.md`, más la procedencia de captura.

    Los nueve primeros campos son los que `types.Sample` exige y los que hacen
    posible la validación leave-one-signer-out. Los últimos cuatro son
    procedencia: no describen la seña sino cómo se grabó, y sirven para auditar el
    dataset sin volver a procesarlo — qué criterio la aceptó, con cuánta σ, y si
    hay un video asociado.
    """

    label: str
    signer_id: str
    session_id: str
    timestamp: datetime
    handedness: Handedness
    light_level: LightLevel
    light_direction: LightDirection
    distance: Distance
    mean_luminance: float
    mean_scale_px: float
    kind: SampleKind
    #: σ del `feature-spec.md` §2 en el momento de aceptar. Se guarda aunque no
    #: haya decidido nada (en dinámicas no decide): al mirar la primera matriz de
    #: confusión sirve para preguntar si las muestras peores eran las más
    #: temblorosas, y esa pregunta no se puede hacer si el número no se guardó.
    dispersion: float
    #: Longitud de arco de τ, en unidades de mano. El espejo de `dispersion`: es
    #: lo que decide en las dinámicas y lo que se guarda sin decidir en las
    #: estáticas.
    arc_length: float
    #: Valor **efectivo** de `hands.mediapipe_reports_mirrored_handedness` al
    #: grabar esta muestra.
    #:
    #: Sin este campo, cambiar el interruptor a mitad del proyecto dejaría el
    #: dataset mezclado sin ningún síntoma: la mitad de las muestras canonizadas
    #: hacia una mano y la otra mitad hacia la contraria, todas con la misma
    #: pinta. Con él, la reparación es un filtro y un espejo en vez de una nueva
    #: ronda de grabaciones.
    handedness_swapped: bool
    #: Qué mano nombra `handedness` (`types.HandednessConvention`). Es la promesa
    #: semántica; `handedness_swapped` es cómo se llegó a ella.
    handedness_convention: HandednessConvention = HANDEDNESS_CONVENTION
    capture_spec_version: int = CAPTURE_SPEC_VERSION
    #: De dónde sale `handedness`. Ver `HandSource`.
    handedness_source: HandSource = HandSource.DECLARED
    #: Nombre del archivo de video junto a la muestra, o `None`. Solo se rellena
    #: con consentimiento explícito por escrito (`ARQUITECTURA.md` §4.11).
    video: str | None = None
    #: Esquema v4 (Bloque 4): frames del principio del flujo que son el trazo,
    #: sin el reposo que lo cerró. Solo en dinámicas grabadas desde el Bloque 4.
    stroke_frames: int | None = None
    #: Esquema v4: con qué `FEATURE_SPEC_VERSION` se calcularon `dispersion` y
    #: `arc_length`. `None` en las muestras anteriores, que no lo anotaban: su σ
    #: no se puede comparar con la de otra versión del contrato.
    feature_spec_version: int | None = None


@dataclass(frozen=True, slots=True)
class StoredSample:
    """Una muestra tal como está en disco: metadatos más el flujo crudo.

    No es `types.Sample` y la diferencia importa. `Sample` lleva una `Sequence`,
    que por construcción no tiene huecos; esto lleva un `FrameStream`, que puede
    tenerlos. Es el registro de lo que pasó frente a la cámara, y de ahí se deriva
    la muestra utilizable — no al revés.
    """

    metadata: SampleMetadata
    frames: FrameStream

    def to_sample(self, preprocessing: Preprocessing | None = None) -> Sample:
        """Convierte a `types.Sample`, que es lo que consume el entrenamiento.

        Exige que el flujo sea **una sola secuencia válida sin huecos**. No es
        pedantería: coser dos trozos separados por un hueco inventaría un
        movimiento entre dos posiciones que nunca se observó, y quedarse con el
        trozo más largo entregaría una seña recortada con la etiqueta de la
        completa. Las dos cosas envenenan el entrenamiento en silencio, así que se
        prefiere fallar aquí, al cargar, donde todavía se puede regrabar.

        **La excepción del Bloque 2** (ADR 0021): con `preprocessing`, la
        muestra pasa por lo mismo que un trazo en vivo (`lsm.preprocessing`): el
        filtro de plausibilidad (§0.4) y, en una **dinámica**, el relleno de
        huecos cortos. Sin él, o en una estática, cualquier hueco sigue siendo
        un error.
        """
        # Bloque 4: lo que se entrena de una dinámica es el trazo, sin el reposo
        # que lo cerró; el reposo queda en el archivo para reproducirla.
        flujo = (
            self.frames[: self.metadata.stroke_frames]
            if self.metadata.stroke_frames is not None
            else self.frames
        )
        runs = split_valid_runs(flujo)
        interpolados = 0
        implausibles = 0
        if preprocessing is not None:
            reconstruida = reconstruct(flujo, self.metadata.kind, preprocessing)
            if isinstance(reconstruida, NotReconstructed):
                msg = (
                    f"la muestra {self.metadata.kind.value.lower()} "
                    f"{self.metadata.label} de {self.metadata.signer_id} tiene "
                    f"huecos que no se pueden rellenar ({reconstruida.reason}: "
                    f"{reconstruida.detail}); hay que regrabarla."
                )
                raise DatasetError(msg)
            secuencia = reconstruida.sequence
            interpolados = reconstruida.interpolated
            implausibles = reconstruida.implausible
        elif len(runs) == 1 and len(runs[0]) == len(flujo):
            secuencia = runs[0]
        else:
            msg = (
                f"la muestra {self.metadata.label} de {self.metadata.signer_id} "
                f"tiene huecos: {len(flujo)} frames en {len(runs)} secuencias "
                "válidas. Una muestra interrumpida no se puede coser ni recortar "
                "sin mentir; hay que regrabarla."
            )
            raise DatasetError(msg)

        meta = self.metadata
        # La mano de la muestra gobierna a todos sus frames: es la declarada
        # (`feature-spec.md` §1, paso 2, v2). Lo que dijo el detector se conserva
        # en `detected_handedness` y no decide nada.
        declarada = Sequence(
            frames=tuple(
                replace(
                    frame,
                    handedness=meta.handedness,
                    detected_handedness=frame.detected_handedness or frame.handedness,
                )
                for frame in secuencia.frames
            )
        )
        return Sample(
            sequence=declarada,
            label=meta.label,
            signer_id=meta.signer_id,
            session_id=meta.session_id,
            timestamp=meta.timestamp,
            handedness=meta.handedness,
            light_level=meta.light_level,
            light_direction=meta.light_direction,
            distance=meta.distance,
            mean_luminance=meta.mean_luminance,
            mean_scale_px=meta.mean_scale_px,
            kind=meta.kind,
            interpolated_frames=interpolados,
            implausible_frames=implausibles,
        )


# --------------------------------------------------------------------------- #
# Disposición en disco
# --------------------------------------------------------------------------- #


def label_dir(root: Path, signer_id: str, session_id: str, label: str) -> Path:
    """`data/raw/<signer>/<sesion>/<LABEL>/`.

    La jerarquía empieza por la persona porque la pregunta que más se hace sobre
    este dataset es "¿qué grabó cada quién?": el split es leave-one-signer-out y
    el borrado a petición de quien firma es por persona. Con las letras arriba,
    las dos operaciones serían un recorrido del árbol entero.

    La ruta duplica lo que ya dice el archivo. Es deliberado y el archivo manda:
    la ruta sirve para navegar y contar sin abrir nada, pero si alguien mueve una
    carpeta, la verdad sigue estando dentro.
    """
    check_identifier("signer_id", signer_id)
    check_identifier("session_id", session_id)
    check_identifier("label", label)
    return root / signer_id / session_id / label


def next_sample_index(directory: Path) -> int:
    """El siguiente número libre de muestra en una carpeta de letra.

    Se toma el máximo existente más uno, y no la cuenta de archivos: si alguien
    borra la muestra 003 por mala, la siguiente debe ser la 006 y no reutilizar un
    número que quizá aparezca en la bitácora de la sesión.
    """
    if not directory.is_dir():
        return 1
    usados = [
        int(path.stem) for path in directory.glob("*.json") if path.stem.isdigit()
    ]
    return max(usados, default=0) + 1


def sample_path(directory: Path, index: int) -> Path:
    """`.../<LABEL>/007.json`. Cero a la izquierda para que ordene bien."""
    return directory / f"{index:03d}.json"


def write_sample(root: Path, sample: StoredSample, index: int | None = None) -> Path:
    """Escribe una muestra y devuelve su ruta.

    `index` existe para quien necesita saber el número **antes** de escribir: el
    nombre del archivo de video va dentro de los metadatos, así que la captura con
    video tiene que reservar el número, construir los metadatos con él y solo
    entonces guardar. Omitirlo toma el siguiente libre, que es lo normal.

    Escribe primero a un archivo temporal y luego renombra. Un `Ctrl-C` a mitad de
    sesión es normal; un JSON truncado en el dataset, descubierto al entrenar tres
    semanas después, no debería serlo.
    """
    meta = sample.metadata
    directory = label_dir(root, meta.signer_id, meta.session_id, meta.label)
    directory.mkdir(parents=True, exist_ok=True)
    destino = sample_path(
        directory, next_sample_index(directory) if index is None else index
    )

    payload = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "capture_spec_version": meta.capture_spec_version,
        "label": meta.label,
        "signer_id": meta.signer_id,
        "session_id": meta.session_id,
        "timestamp": meta.timestamp.isoformat(),
        "handedness": str(meta.handedness),
        "light_level": str(meta.light_level),
        "light_direction": str(meta.light_direction),
        "distance": str(meta.distance),
        "mean_luminance": meta.mean_luminance,
        "mean_scale_px": meta.mean_scale_px,
        "kind": str(meta.kind),
        "dispersion": meta.dispersion,
        "arc_length": meta.arc_length,
        "handedness_swapped": meta.handedness_swapped,
        "handedness_convention": str(meta.handedness_convention),
        "handedness_source": str(meta.handedness_source),
        "video": meta.video,
        "stroke_frames": meta.stroke_frames,
        "feature_spec_version": meta.feature_spec_version,
        "frames": frames_to_json(sample.frames),
    }

    temporal = destino.with_suffix(".json.tmp")
    temporal.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporal.replace(destino)
    return destino


def read_sample(path: Path) -> StoredSample:
    """Lee una muestra. Rechaza los archivos de otra versión de esquema."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version not in READABLE_SAMPLE_SCHEMAS:
        msg = (
            f"{path}: schema_version {version} incompatible; este código lee las "
            f"versiones {sorted(READABLE_SAMPLE_SCHEMAS)}"
        )
        raise DatasetError(msg)

    timestamp = datetime.fromisoformat(payload["timestamp"])
    if timestamp.tzinfo is None:
        msg = f"{path}: el timestamp no lleva zona horaria; no es comparable"
        raise DatasetError(msg)

    metadata = SampleMetadata(
        label=payload["label"],
        signer_id=payload["signer_id"],
        session_id=payload["session_id"],
        timestamp=timestamp,
        handedness=Handedness(payload["handedness"]),
        light_level=LightLevel(payload["light_level"]),
        light_direction=LightDirection(payload["light_direction"]),
        distance=Distance(payload["distance"]),
        mean_luminance=payload["mean_luminance"],
        mean_scale_px=payload["mean_scale_px"],
        kind=SampleKind(payload["kind"]),
        dispersion=payload["dispersion"],
        arc_length=payload["arc_length"],
        handedness_swapped=bool(payload["handedness_swapped"]),
        handedness_convention=HandednessConvention(payload["handedness_convention"]),
        capture_spec_version=payload["capture_spec_version"],
        handedness_source=(
            HandSource(payload["handedness_source"])
            if version >= 3
            else HandSource.DETECTED
        ),
        video=payload.get("video"),
        stroke_frames=payload.get("stroke_frames") if version >= 4 else None,
        feature_spec_version=(
            payload.get("feature_spec_version") if version >= 4 else None
        ),
    )
    return StoredSample(metadata=metadata, frames=frames_from_json(payload["frames"]))


def iter_sample_paths(root: Path) -> Iterator[Path]:
    """Todas las muestras del dataset, en orden estable.

    Ordenado a propósito: un recorrido cuyo orden depende del sistema de archivos
    hace que dos ejecuciones de la evaluación no sean comparables entre sí.

    **Las sesiones de prueba quedan fuera.** Viven un nivel más abajo, en
    `pruebas/`, así que el patrón ya no las alcanzaría; se excluyen además de
    forma explícita porque depender de que un glob tenga la profundidad justa es
    la clase de garantía que se rompe la primera vez que alguien reorganiza una
    carpeta, y el síntoma sería un modelo entrenado con las grabaciones que se
    hicieron para probar el enfoque de la cámara.

    Para recorrerlas, se apunta la raíz a ellas: `--raiz data/raw/pruebas`.
    """
    if not root.is_dir():
        return
    yield from sorted(
        path
        for path in root.glob("*/*/*/[0-9]*.json")
        # Relativo a la raíz, no absoluto: apuntar la raíz **a** `pruebas/` es la
        # forma soportada de recorrerlas (`--raiz data/raw/pruebas`), y con una
        # comprobación sobre la ruta completa esa raíz se excluiría a sí misma.
        # Los intentos rechazados (Paso 0) quedan un nivel más abajo y el patrón
        # no los alcanza; se excluyen también por nombre, por lo mismo.
        if TRIAL_DIRNAME not in (partes := path.relative_to(root).parts)
        and REJECTED_DIRNAME not in partes
    )


def trial_root(root: Path) -> Path:
    """Dónde escriben las sesiones de prueba (`--sesion-prueba`)."""
    return root / TRIAL_DIRNAME


def iter_samples(root: Path) -> Iterator[StoredSample]:
    for path in iter_sample_paths(root):
        yield read_sample(path)


def count_samples(root: Path, signer_id: str, session_id: str) -> dict[str, int]:
    """Cuántas muestras hay por letra en una sesión.

    Cuenta archivos sin abrirlos: alimenta el contador del preview, que se
    actualiza treinta veces por segundo y no puede permitirse leer el dataset.
    """
    sesion = root / signer_id / session_id
    if not sesion.is_dir():
        return {}
    return {
        carpeta.name: sum(1 for path in carpeta.glob("*.json") if path.stem.isdigit())
        for carpeta in sorted(sesion.iterdir())
        if carpeta.is_dir()
    }


# --------------------------------------------------------------------------- #
# Consentimiento
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Consent:
    """Lo que quien firma autorizó, y cuándo.

    Los landmarks no identifican a nadie; el video sí, y por eso las dos cosas se
    autorizan por separado. `video` en `False` no impide grabar el dataset: impide
    guardar cuadros de la persona, que es lo que hay que pedir permiso para
    guardar (`ARQUITECTURA.md` §4.11).

    Este archivo **no es** el consentimiento: el consentimiento es por escrito y
    en papel o su equivalente. Esto es el registro de que existe, y `referencia`
    es dónde encontrarlo.
    """

    signer_id: str
    #: Autoriza almacenar cuadros de video, no solo landmarks.
    video: bool
    #: Fecha del registro.
    fecha: datetime
    #: Dónde está el consentimiento firmado: folio, archivo, expediente.
    referencia: str = ""


def consent_path(root: Path) -> Path:
    return root / CONSENT_FILENAME


def load_consents(root: Path) -> dict[str, Consent]:
    """Lee el registro. Un dataset sin registro es un dataset sin permisos."""
    path = consent_path(root)
    if not path.is_file():
        return {}

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != CONSENT_SCHEMA_VERSION:
        msg = (
            f"{path}: schema_version {version} incompatible; este código lee la "
            f"versión {CONSENT_SCHEMA_VERSION}"
        )
        raise DatasetError(msg)

    return {
        signer_id: Consent(
            signer_id=signer_id,
            video=bool(entry["video"]),
            fecha=datetime.fromisoformat(entry["fecha"]),
            referencia=entry.get("referencia", ""),
        )
        for signer_id, entry in payload.get("firmantes", {}).items()
    }


def save_consent(root: Path, consent: Consent) -> Path:
    """Registra o actualiza el consentimiento de una persona."""
    check_identifier("signer_id", consent.signer_id)
    registro = load_consents(root)
    registro[consent.signer_id] = consent

    root.mkdir(parents=True, exist_ok=True)
    path = consent_path(root)
    payload = {
        "schema_version": CONSENT_SCHEMA_VERSION,
        "firmantes": {
            signer_id: {
                "video": entry.video,
                "fecha": entry.fecha.isoformat(),
                "referencia": entry.referencia,
            }
            for signer_id, entry in sorted(registro.items())
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def may_store_video(root: Path, signer_id: str) -> bool:
    """Si el registro autoriza guardar video de esta persona.

    Devuelve `False` cuando no hay registro. El valor por defecto de una pregunta
    sobre permisos es "no": un archivo que falta es una autorización que no se
    pidió, nunca una que se dio.
    """
    consent = load_consents(root).get(signer_id)
    return consent is not None and consent.video


def now() -> datetime:
    """Instante actual **con zona horaria**.

    `Sample` rechaza los timestamps ingenuos: sin offset, dos sesiones grabadas en
    husos distintos no se pueden ordenar, y ordenar sesiones es lo primero que se
    hace al depurar por qué una tanda salió peor que otra.
    """
    return datetime.now(UTC).astimezone()


# --------------------------------------------------------------------------- #
# Manifiesto de dinámicas truncadas (Bloque 4)
# --------------------------------------------------------------------------- #


def write_truncated_manifest(
    root: Path, entries: dict[str, float], criterion: dict[str, Any]
) -> Path:
    """Escribe el manifiesto: ruta relativa → velocidad final medida.

    Se reescribe entero cada vez: el criterio es determinista sobre los mismos
    archivos, así que volver a marcar da lo mismo.
    """
    destino = root / TRUNCATED_MANIFEST
    payload = {
        "schema_version": TRUNCATED_MANIFEST_VERSION,
        "criterion": criterion,
        "samples": [
            {"path": ruta, "final_velocity_per_s": velocidad}
            for ruta, velocidad in sorted(entries.items())
        ],
    }
    temporal = destino.with_suffix(".json.tmp")
    temporal.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporal.replace(destino)
    return destino


def read_truncated_manifest(root: Path) -> frozenset[str]:
    """Las rutas relativas marcadas como truncadas, o ninguna si no hay manifiesto."""
    ruta = root / TRUNCATED_MANIFEST
    if not ruta.exists():
        return frozenset()
    payload: Any = json.loads(ruta.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRUNCATED_MANIFEST_VERSION:
        msg = (
            f"{ruta}: schema_version {payload.get('schema_version')} incompatible; "
            f"este código lee la {TRUNCATED_MANIFEST_VERSION}"
        )
        raise DatasetError(msg)
    return frozenset(entrada["path"] for entrada in payload["samples"])


# --------------------------------------------------------------------------- #
# Intentos dinámicos rechazados (Paso 0)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RejectedAttempt:
    """Una grabación dinámica que la captura rechazó, entera y en crudo.

    Se guarda **todo** lo que la cámara entregó desde que ESPACIO armó la
    grabación hasta que se rechazó —huecos incluidos—, con el motivo. No es una
    muestra: no tiene trazo delimitado ni σ, y el entrenamiento no la lee. Existe
    para que, cuando mejore la tolerancia de la tubería, el intento se pueda
    volver a segmentar (`lsm.capture.resegment_attempt`) sin regrabarlo.
    """

    label: str
    signer_id: str
    session_id: str
    timestamp: datetime
    #: La mano declarada de la sesión, igual que en una muestra.
    handedness: Handedness
    light_level: LightLevel
    light_direction: LightDirection
    distance: Distance
    mean_luminance: float
    #: `lsm.capture.Rejection`, como texto.
    reason: str
    #: La tasa con la que la máquina de estados convirtió sus umbrales al
    #: delimitar: re-segmentar con otra daría otros cortes.
    fps: float
    #: La sección `segmentation` de la configuración con que se rechazó, para
    #: saber contra qué umbrales perdió.
    segmentation: dict[str, Any]
    frames: FrameStream
    capture_spec_version: int = CAPTURE_SPEC_VERSION
    segmentation_spec_version: int = SEGMENTATION_SPEC_VERSION


def rejected_dir(root: Path, signer_id: str, session_id: str, label: str) -> Path:
    """`data/raw/<signer>/<sesion>/<LABEL>/rechazados/`."""
    return label_dir(root, signer_id, session_id, label) / REJECTED_DIRNAME


def write_rejected_attempt(root: Path, attempt: RejectedAttempt) -> Path:
    """Escribe un intento rechazado con el siguiente número libre de su carpeta."""
    directory = rejected_dir(root, attempt.signer_id, attempt.session_id, attempt.label)
    directory.mkdir(parents=True, exist_ok=True)
    destino = sample_path(directory, next_sample_index(directory))
    payload = {
        "schema_version": REJECTED_ATTEMPT_SCHEMA_VERSION,
        "kind": "REJECTED_ATTEMPT",
        "capture_spec_version": attempt.capture_spec_version,
        "segmentation_spec_version": attempt.segmentation_spec_version,
        "label": attempt.label,
        "signer_id": attempt.signer_id,
        "session_id": attempt.session_id,
        "timestamp": attempt.timestamp.isoformat(),
        "handedness": str(attempt.handedness),
        "light_level": str(attempt.light_level),
        "light_direction": str(attempt.light_direction),
        "distance": str(attempt.distance),
        "mean_luminance": attempt.mean_luminance,
        "rejection": attempt.reason,
        "fps": attempt.fps,
        "segmentation": attempt.segmentation,
        "frames": frames_to_json(attempt.frames),
    }
    temporal = destino.with_suffix(".json.tmp")
    temporal.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporal.replace(destino)
    return destino


def read_rejected_attempt(path: Path) -> RejectedAttempt:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if payload.get("kind") != "REJECTED_ATTEMPT" or (
        version != REJECTED_ATTEMPT_SCHEMA_VERSION
    ):
        msg = (
            f"{path}: no es un intento rechazado de la versión "
            f"{REJECTED_ATTEMPT_SCHEMA_VERSION} (kind {payload.get('kind')!r}, "
            f"schema_version {version})"
        )
        raise DatasetError(msg)
    return RejectedAttempt(
        label=payload["label"],
        signer_id=payload["signer_id"],
        session_id=payload["session_id"],
        timestamp=datetime.fromisoformat(payload["timestamp"]),
        handedness=Handedness(payload["handedness"]),
        light_level=LightLevel(payload["light_level"]),
        light_direction=LightDirection(payload["light_direction"]),
        distance=Distance(payload["distance"]),
        mean_luminance=payload["mean_luminance"],
        reason=payload["rejection"],
        fps=payload["fps"],
        segmentation=payload["segmentation"],
        frames=frames_from_json(payload["frames"]),
        capture_spec_version=payload["capture_spec_version"],
        segmentation_spec_version=payload["segmentation_spec_version"],
    )


def iter_rejected_attempt_paths(root: Path) -> Iterator[Path]:
    """Todos los intentos rechazados, en orden estable, sin los de prueba."""
    if not root.is_dir():
        return
    yield from sorted(
        path
        for path in root.glob(f"*/*/*/{REJECTED_DIRNAME}/[0-9]*.json")
        if TRIAL_DIRNAME not in path.relative_to(root).parts
    )
