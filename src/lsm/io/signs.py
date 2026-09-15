"""Texto → señas: lo que toca disco y Pillow.

`lsm.signs` decide; esto lee el manifest, carga candidatas de `data/raw`, dibuja
los esqueletos y los guarda. Pillow y no OpenCV porque OpenCV no escribe GIF y
su fuente es ASCII: las descripciones del glosario llevan acentos y una Ñ.

Pillow es dependencia del extra `capture` y del grupo `dev`, no del núcleo.
Se importa arriba porque este módulo entero es I/O; nadie del núcleo lo importa.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw, ImageFont

from lsm.config import Config
from lsm.io.dataset import DatasetError, iter_sample_paths, read_sample
from lsm.signs import Candidate, Manifest, Scene, WordGap
from lsm.types import HAND_CONNECTIONS, LandmarkIndex, Point2
from lsm.vocabulary import LETTERS, Label

_logger = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------- #
# El cuadro de la ventana
# --------------------------------------------------------------------------- #

_PANEL_ANCHO: Final = 520
_PIE_ALTO: Final = 96
_MARGEN_TEXTO: Final = 24
_TINTA: Final = (30, 30, 30)
_TINTA_SUAVE: Final = (110, 110, 110)
_ACENTO: Final = (30, 80, 200)
_BARRA_FONDO: Final = (225, 225, 222)
_AVISO: Final = "Deletreo manual, no LSM como lengua. Procesamiento local."
_ATAJOS: Final = (
    "ESPACIO pausa | n/p siguiente/anterior | r reinicio | +/- velocidad | q salir"
)

#: Empaquetada en `io/fonts/` (ver el README ahí): la fuente por defecto de
#: Pillow (Aileron) no tiene glifos de Ñ/ñ ni vocales acentuadas y los dibuja
#: como recuadros. DejaVu Sans sí los tiene, y viaja con el paquete para que
#: el cuadro se vea igual en cualquier entorno (CI, Docker, Windows).
_FONT_PATH: Final = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"


@functools.cache
def _fuente(tamano: int) -> Any:
    # `functools.cache` evita reabrir y reparsear el archivo en cada símbolo de
    # cada cuadro: `draw_scene` pide la misma fuente ~10 veces por llamada.
    try:
        return ImageFont.truetype(str(_FONT_PATH), tamano)
    except OSError:
        _logger.warning(
            "no se encontró la fuente empaquetada %s; usando la fuente por "
            "defecto de Pillow, que no dibuja Ñ ni acentos",
            _FONT_PATH,
        )
        return ImageFont.load_default(size=tamano)


def _envolver(
    texto: str, ancho_max: int, fuente: Any, lapiz: ImageDraw.ImageDraw
) -> list[str]:
    lineas: list[str] = []
    actual = ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if lapiz.textlength(prueba, font=fuente) <= ancho_max or not actual:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


def _ventana_de_simbolos(
    anchos: SequenceABC[int], actual: int, disponible: int
) -> tuple[int, int]:
    """Un rango `[inicio, fin)` de símbolos que cabe en `disponible` píxeles,
    contiene `actual` y lo deja lo más centrado posible.

    Parte de una ventana de un solo símbolo (`actual`) y la expande alternando
    izquierda y derecha mientras siga cabiendo; cuando un lado ya no cabe (o se
    acaban los símbolos de ese lado) se abandona ese lado y se sigue intentando
    con el otro, hasta que ninguno de los dos puede crecer más. Con un texto
    corto, eso agota los dos lados en los extremos y el rango es el completo.
    """
    n = len(anchos)
    inicio = fin = actual
    total = anchos[actual]
    fin += 1
    izquierda_agotada = inicio == 0
    derecha_agotada = fin == n
    turno_izquierda = True
    while not (izquierda_agotada and derecha_agotada):
        if turno_izquierda and not izquierda_agotada:
            candidato = anchos[inicio - 1]
            if total + candidato <= disponible:
                total += candidato
                inicio -= 1
                izquierda_agotada = inicio == 0
            else:
                izquierda_agotada = True
        elif not turno_izquierda and not derecha_agotada:
            candidato = anchos[fin]
            if total + candidato <= disponible:
                total += candidato
                fin += 1
                derecha_agotada = fin == n
            else:
                derecha_agotada = True
        turno_izquierda = not turno_izquierda
    return inicio, fin


def draw_scene(scene: Scene, config: Config) -> Image.Image:
    """Asset a la izquierda; letra, descripción, progreso y estado a la derecha;
    el texto completo con el símbolo actual resaltado abajo."""
    lado = config.signs.canvas_px
    ancho, alto = lado + _PANEL_ANCHO, lado + _PIE_ALTO
    cuadro = Image.new("RGB", (ancho, alto), _FONDO)
    lapiz = ImageDraw.Draw(cuadro)

    # Panel izquierdo: el asset, o "espacio" en una pausa.
    if scene.frame is not None:
        cuadro.paste(scene.frame.resize((lado, lado)), (0, 0))
    else:
        lapiz.rectangle([0, 0, lado, lado], fill=_BARRA_FONDO)
        lapiz.text(
            (lado // 2, lado // 2),
            "espacio",
            fill=_TINTA_SUAVE,
            font=_fuente(28),
            anchor="mm",
        )

    # Panel derecho.
    x = lado + _MARGEN_TEXTO
    y = _MARGEN_TEXTO
    posicion = f"{scene.state.index + 1} / {len(scene.playlist)}"
    lapiz.text(
        (ancho - _MARGEN_TEXTO, y),
        posicion,
        fill=_TINTA_SUAVE,
        font=_fuente(20),
        anchor="ra",
    )
    if scene.asset is not None:
        lapiz.text((x, y), scene.asset.letra, fill=_TINTA, font=_fuente(72))
        y += 96
        for linea in _envolver(
            scene.asset.descripcion,
            _PANEL_ANCHO - 2 * _MARGEN_TEXTO,
            _fuente(18),
            lapiz,
        ):
            lapiz.text((x, y), linea, fill=_TINTA, font=_fuente(18))
            y += 24
        if scene.asset.es_dinamica:
            y += 8
            lapiz.text(
                (x, y),
                f"Trayectoria: {scene.asset.trayectoria}",
                fill=_ACENTO,
                font=_fuente(18),
            )
            y += 24
    else:
        lapiz.text((x, y), "pausa entre palabras", fill=_TINTA_SUAVE, font=_fuente(28))

    # Estado, velocidad y barra de progreso, pegados al borde inferior del panel.
    estado = (
        "PAUSA"
        if scene.state.paused
        else ("FIN" if scene.state.finished else "REPRODUCIENDO")
    )
    y_barra = lado - _MARGEN_TEXTO - 32
    lapiz.text(
        (x, y_barra - 28),
        f"{estado}   {scene.state.speed:.2f}x",
        fill=_TINTA,
        font=_fuente(18),
    )
    ancho_barra = _PANEL_ANCHO - 2 * _MARGEN_TEXTO
    lapiz.rectangle([x, y_barra, x + ancho_barra, y_barra + 12], fill=_BARRA_FONDO)
    lapiz.rectangle(
        [x, y_barra, x + int(ancho_barra * scene.progress), y_barra + 12], fill=_ACENTO
    )
    transcurrido = scene.state.elapsed_ms / 1000.0
    total = scene.step.duration_ms / 1000.0
    lapiz.text(
        (x + ancho_barra, y_barra + 16),
        f"{transcurrido:.1f} s / {total:.1f} s",
        fill=_TINTA_SUAVE,
        font=_fuente(16),
        anchor="ra",
    )

    # Pie: el texto completo, símbolo a símbolo, con el actual resaltado. Un
    # texto largo no cabe entero: `_ventana_de_simbolos` elige un rango
    # contiguo que sí cabe y que mantiene el símbolo actual centrado; "…" en
    # los bordes avisa que hay más texto fuera de cuadro.
    y_pie = lado + 20
    fuente_pie = _fuente(26)
    simbolos = [
        "·" if isinstance(token, WordGap) else LETTERS[token].display
        for token in scene.tokens
    ]
    anchos = [
        int(lapiz.textlength(simbolo + "  ", font=fuente_pie)) for simbolo in simbolos
    ]
    elipsis_ancho = int(lapiz.textlength("… ", font=fuente_pie))
    disponible = ancho - 2 * _MARGEN_TEXTO - 2 * elipsis_ancho
    inicio, fin = _ventana_de_simbolos(anchos, scene.token_index, disponible)

    x_pie = _MARGEN_TEXTO
    if inicio > 0:
        lapiz.text((x_pie, y_pie), "… ", fill=_TINTA_SUAVE, font=fuente_pie)
        x_pie += elipsis_ancho
    for indice in range(inicio, fin):
        color = _ACENTO if indice == scene.token_index else _TINTA
        lapiz.text((x_pie, y_pie), simbolos[indice], fill=color, font=fuente_pie)
        x_pie += anchos[indice]
    if fin < len(simbolos):
        lapiz.text((x_pie, y_pie), "…", fill=_TINTA_SUAVE, font=fuente_pie)
    lapiz.text((_MARGEN_TEXTO, alto - 40), _ATAJOS, fill=_TINTA_SUAVE, font=_fuente(14))
    lapiz.text((_MARGEN_TEXTO, alto - 22), _AVISO, fill=_TINTA_SUAVE, font=_fuente(14))
    return cuadro
