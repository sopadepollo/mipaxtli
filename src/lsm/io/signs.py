"""Texto → señas: lo que toca disco y Pillow.

`lsm.signs` decide; esto lee el manifest, carga candidatas de `data/raw`, dibuja
los esqueletos y los guarda. Pillow y no OpenCV porque OpenCV no escribe GIF y
su fuente es ASCII: las descripciones del glosario llevan acentos y una Ñ.

Pillow es dependencia del extra `capture` y del grupo `dev`, no del núcleo.
Se importa arriba porque este módulo entero es I/O; nadie del núcleo lo importa.
"""

from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw

from lsm.io.dataset import DatasetError, iter_sample_paths, read_sample
from lsm.signs import Candidate, Manifest
from lsm.types import HAND_CONNECTIONS, LandmarkIndex, Point2
from lsm.vocabulary import Label

MANIFEST_FILENAME: Final = "manifest.json"
DEFAULT_ASSETS_DIR: Final = Path("assets/signs")
DEFAULT_RAW_DIR: Final = Path("data/raw")

# Constantes de dibujo, no umbrales: nada de esto cambia lo que el sistema
# decide, solo cómo se ve.
_FONDO: Final = (250, 250, 248)
_HUESO: Final = (40, 110, 60)
_ARTICULACION: Final = (25, 25, 25)
_YEMA: Final = (200, 60, 40)
_MUNECA: Final = (30, 80, 200)
_GROSOR_HUESO: Final = 4
_RADIO_ARTICULACION: Final = 4
_RADIO_YEMA: Final = 6
_RADIO_MUNECA: Final = 7
_YEMAS: Final = frozenset(
    {
        LandmarkIndex.THUMB_TIP,
        LandmarkIndex.INDEX_TIP,
        LandmarkIndex.MIDDLE_TIP,
        LandmarkIndex.RING_TIP,
        LandmarkIndex.PINKY_TIP,
    }
)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def load_manifest(path: Path) -> Manifest:
    """Lee y valida. Los errores de esquema salen como `ValidationError`."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(payload)


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Escribe con claves ordenadas y salto final: mismos datos, mismos bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Candidatas
# --------------------------------------------------------------------------- #


def load_candidates(root: Path) -> dict[Label, list[Candidate]]:
    """Todas las muestras utilizables del dataset, agrupadas por letra.

    Una muestra con huecos (`to_sample` la rechaza) o con una etiqueta que el
    vocabulario no conoce se salta: no sirve de referencia y no es motivo para
    no renderizar las demás.
    """
    por_letra: dict[Label, list[Candidate]] = {}
    for ruta in iter_sample_paths(root):
        stored = read_sample(ruta)
        try:
            label = Label(stored.metadata.label)
            sample = stored.to_sample()
        except (ValueError, DatasetError):
            continue
        por_letra.setdefault(label, []).append(
            Candidate(sample=sample, path=ruta.relative_to(root).as_posix())
        )
    return por_letra


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #


def draw_skeleton(points: SequenceABC[Point2], canvas_px: int) -> Image.Image:
    """Un frame: huesos, articulaciones, yemas y muñeca destacadas."""
    imagen = Image.new("RGB", (canvas_px, canvas_px), _FONDO)
    lapiz = ImageDraw.Draw(imagen)
    for inicio, fin in HAND_CONNECTIONS:
        lapiz.line([points[inicio], points[fin]], fill=_HUESO, width=_GROSOR_HUESO)
    for indice, (x, y) in enumerate(points):
        if indice == LandmarkIndex.WRIST:
            radio, color = _RADIO_MUNECA, _MUNECA
        elif indice in _YEMAS:
            radio, color = _RADIO_YEMA, _YEMA
        else:
            radio, color = _RADIO_ARTICULACION, _ARTICULACION
        lapiz.ellipse([x - radio, y - radio, x + radio, y + radio], fill=color)
    return imagen


def write_static_asset(
    path: Path, frames_points: SequenceABC[SequenceABC[Point2]], canvas_px: int
) -> None:
    """PNG del frame central de la secuencia."""
    central = frames_points[len(frames_points) // 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_skeleton(central, canvas_px).save(path, format="PNG", optimize=True)


def write_dynamic_asset(
    path: Path,
    frames_points: SequenceABC[SequenceABC[Point2]],
    canvas_px: int,
    fps: int,
) -> int:
    """GIF con todos los frames, en bucle. Devuelve `duracion_ms` de una vuelta."""
    cuadros = [draw_skeleton(puntos, canvas_px) for puntos in frames_points]
    path.parent.mkdir(parents=True, exist_ok=True)
    cuadros[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=cuadros[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return round(1000 * len(cuadros) / fps)


def gif_frame_count(path: Path) -> int:
    with Image.open(path) as imagen:
        return int(getattr(imagen, "n_frames", 1))


def load_asset_frames(path: Path) -> list[Image.Image]:
    """Todos los cuadros de un asset como imágenes RGB independientes."""
    cuadros: list[Image.Image] = []
    with Image.open(path) as imagen:
        for indice in range(int(getattr(imagen, "n_frames", 1))):
            imagen.seek(indice)
            cuadros.append(imagen.convert("RGB").copy())
    return cuadros
