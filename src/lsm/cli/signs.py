"""`lsm-signs`: la dirección texto → señas (`ARQUITECTURA.md` §4.10, Fase 4).

Tres subcomandos:

- `render`: dibuja los 29 assets desde `data/raw` y escribe el manifest.
- `verificar`: el manifest y los archivos están completos y revisados.
- `reproducir "texto"`: la ventana, con temporizador y control manual.

No hay cámara en ninguno. OpenCV solo hace falta en `reproducir`, para la
ventana, y se importa ahí dentro.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import ValidationError

from lsm.cli import MENSAJE_SIN_EXTRAS
from lsm.config import Config, load_config
from lsm.io.signs import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_RAW_DIR,
    MANIFEST_FILENAME,
    draw_scene,
    gif_frame_count,
    load_asset_frames,
    load_candidates,
    load_manifest,
    save_manifest,
    write_dynamic_asset,
    write_static_asset,
)
from lsm.signs import (
    FUENTE_NORMATIVA,
    MANIFEST_SCHEMA_VERSION,
    AssetReview,
    AssetSource,
    Faster,
    Manifest,
    Next,
    PlayerInput,
    PlayerState,
    Prev,
    ReferenceChoice,
    Restart,
    Scene,
    SignAsset,
    Slower,
    Step,
    Tick,
    TogglePause,
    Token,
    UnsupportedCharacters,
    WordGap,
    asset_frame,
    build_playlist,
    choose_reference,
    expected_filename,
    manifest_drift,
    player_step,
    project_frames,
    render_tokens,
    start,
    text_to_symbols,
)
from lsm.vocabulary import ALPHABET, LETTERS, Label

SIN_MANIFEST: Final = (
    "no existe {ruta}.\n"
    "Los assets se generan desde el dataset propio:\n"
    "  lsm-signs render --revisor <tu nombre>"
)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


def render(raw: Path, assets: Path, config: Config, revisor: str, hoy: date) -> int:
    """Dibuja un asset por letra y escribe el manifest. Falla antes de escribir
    nada si a alguna letra le faltan candidatas válidas (incluida una letra sin
    ninguna muestra) o si el manifest previo no se puede leer."""
    candidatas = load_candidates(raw)

    # Fase 1: elegir la referencia de las 29 letras antes de tocar disco.
    # `choose_reference` filtra por `kind` (una letra dinámica grabada solo como
    # estática, por ejemplo) y lanza `ValueError` nombrando la letra; se recogen
    # todos los fallos para no escribir nada a medias.
    elecciones: dict[Label, ReferenceChoice] = {}
    errores: list[str] = []
    for label in ALPHABET:
        try:
            elecciones[label] = choose_reference(
                candidatas.get(label, []), label, config
            )
        except ValueError as error:
            errores.append(str(error))
    if errores:
        print("No se escribe nada:")
        for mensaje in errores:
            print(f"  {mensaje}")
        return 1

    ruta_manifest = assets / MANIFEST_FILENAME
    previo: Manifest | None = None
    if ruta_manifest.is_file():
        try:
            previo = load_manifest(ruta_manifest)
        except (ValidationError, json.JSONDecodeError) as error:
            print(
                f"{ruta_manifest} no se puede leer ({error.__class__.__name__}): "
                "corrígelo o bórralo antes de renderizar; sus revisiones se perderían"
            )
            return 1

    # Fase 2: ya se sabe que las 29 letras tienen referencia; dibujar y escribir.
    letras: dict[Label, SignAsset] = {}
    for label in ALPHABET:
        letra = LETTERS[label]
        eleccion = elecciones[label]
        puntos = project_frames(
            eleccion.sample.sequence.frames,
            mirrored=eleccion.mirrored,
            canvas_px=config.signs.canvas_px,
            margin=config.signs.canvas_margin,
        )
        archivo = assets / expected_filename(label)
        duracion: int | None = None
        if letra.es_dinamica:
            duracion = write_dynamic_asset(
                archivo, puntos, config.signs.canvas_px, config.signs.render_fps
            )
        else:
            write_static_asset(archivo, puntos, config.signs.canvas_px)

        anterior = previo.letras.get(label) if previo is not None else None
        if anterior is not None and anterior.fuente.muestra == eleccion.path:
            revision = anterior.revision
        else:
            revision = AssetReview(fecha=hoy, revisor=revisor, resultado="pendiente")

        letras[label] = SignAsset(
            letra=letra.display,
            archivo=archivo.name,
            es_dinamica=letra.es_dinamica,
            descripcion=letra.descripcion,
            trayectoria=letra.trayectoria,
            pagina=letra.pagina,
            duracion_ms=duracion,
            fuente=AssetSource(
                tipo="esqueleto_desde_dataset",
                muestra=eleccion.path,
                signer_id=eleccion.sample.signer_id,
                lateralidad_original=eleccion.sample.handedness,
                espejada=eleccion.mirrored,
            ),
            revision=revision,
        )
        espejada = ", espejada" if eleccion.mirrored else ""
        print(
            f"{letra.display:>3}  {archivo.name:<14} {eleccion.path}  "
            f"({eleccion.candidates} candidatas{espejada})"
        )

    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )
    save_manifest(manifest, ruta_manifest)
    pendientes = sum(1 for a in letras.values() if a.revision.resultado != "coincide")
    print(f"\n{len(letras)} assets en {assets}; {pendientes} revisiones pendientes.")
    return 0


# --------------------------------------------------------------------------- #
# verificar
# --------------------------------------------------------------------------- #


def verificar(assets: Path, config: Config) -> list[str]:
    """Todo lo que tiene que cumplirse para dar la fase por cerrada."""
    ruta_manifest = assets / MANIFEST_FILENAME
    if not ruta_manifest.is_file():
        return [SIN_MANIFEST.format(ruta=ruta_manifest)]
    manifest = load_manifest(ruta_manifest)
    problemas = manifest_drift(manifest)
    for label, asset in manifest.letras.items():
        archivo = assets / asset.archivo
        if not archivo.is_file():
            problemas.append(f"{label}: falta el archivo {asset.archivo}")
        elif asset.es_dinamica:
            assert asset.duracion_ms is not None
            esperados = round(asset.duracion_ms * config.signs.render_fps / 1000)
            reales = gif_frame_count(archivo)
            if reales != esperados:
                problemas.append(
                    f"{label}: {asset.archivo} tiene {reales} frames y duracion_ms "
                    f"dice {esperados}"
                )
        if asset.revision.resultado != "coincide":
            problemas.append(
                f"{label}: revisión {asset.revision.resultado}"
                + (f" — {asset.revision.nota}" if asset.revision.nota else "")
            )
    return problemas


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-signs",
        description=(
            "Deletreo manual, dirección texto → señas: muestra la secuencia de "
            "señas del abecedario LSM para un texto. Sin cámara."
        ),
    )
    # Opciones comunes como `parents`: argparse solo reconoce las opciones del
    # subparser que está parseando, así que `lsm-signs render --assets X` no
    # funcionaría con `--assets` definido solo en el parser raíz.
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--config", type=Path, default=Path("config.yaml"))
    comun.add_argument(
        "--assets",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="carpeta con manifest.json y los PNG/GIF",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_render = sub.add_parser(
        "render", parents=[comun], help="dibuja los assets desde el dataset propio"
    )
    p_render.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    p_render.add_argument(
        "--revisor",
        required=True,
        help="quién queda como revisor de las entradas nuevas (resultado: pendiente)",
    )
    p_render.add_argument(
        "--hoy",
        type=date.fromisoformat,
        default=None,
        help="fecha de las revisiones nuevas; por defecto, hoy",
    )

    sub.add_parser(
        "verificar",
        parents=[comun],
        help="manifest completo, archivos presentes, revisiones cerradas",
    )

    p_play = sub.add_parser(
        "reproducir", parents=[comun], help="muestra la secuencia de señas de un texto"
    )
    p_play.add_argument("texto", help='el texto a deletrear, por ejemplo "casa"')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.comando == "render":
        hoy = args.hoy if args.hoy is not None else date.today()
        return render(args.raw, args.assets, config, args.revisor, hoy)

    if args.comando == "verificar":
        problemas = verificar(args.assets, config)
        for problema in problemas:
            print(problema)
        if problemas:
            print(f"\n{len(problemas)} problema(s).")
            return 1
        print("manifest completo: 29 letras con archivo, fuente y revisión.")
        return 0

    return reproducir(args.texto, args.assets, config)


# --------------------------------------------------------------------------- #
# reproducir
# --------------------------------------------------------------------------- #

_SALIR: Final = frozenset({ord("q"), 27})
_TECLAS: Final[dict[int, PlayerInput]] = {
    ord(" "): TogglePause(),
    ord("n"): Next(),
    ord("p"): Prev(),
    ord("r"): Restart(),
    ord("+"): Faster(),
    ord("="): Faster(),
    ord("-"): Slower(),
}


class Ventana(Protocol):
    """Lo que el bucle necesita de una ventana. La real es OpenCV; los tests
    inyectan una falsa con teclas programadas."""

    def mostrar(self, imagen: Any) -> None: ...
    def tecla(self, espera_ms: int) -> int: ...
    def cerrar(self) -> None: ...


class VentanaOpenCV:
    def __init__(self, titulo: str) -> None:
        self.titulo = titulo

    def mostrar(self, imagen: Any) -> None:
        import cv2
        import numpy as np

        # Pillow entrega RGB; OpenCV espera BGR.
        cv2.imshow(self.titulo, np.asarray(imagen)[:, :, ::-1])

    def tecla(self, espera_ms: int) -> int:
        import cv2

        return int(cv2.waitKey(espera_ms) & 0xFF)

    def cerrar(self) -> None:
        import cv2

        cv2.destroyAllWindows()


def bucle(
    playlist: tuple[Step, ...],
    tokens: tuple[Token, ...],
    manifest: Manifest,
    cuadros: Mapping[Label, list[Any]],
    config: Config,
    ventana: Ventana,
    reloj: Callable[[], float],
    componer: Callable[[Scene], Any],
) -> PlayerState:
    """Mostrar, leer tecla, medir el tiempo, avanzar. Hasta `q`.

    El `dt` de cada `Tick` es tiempo de pared entre vueltas, medido con `reloj`:
    `config.signs.tick_ms` solo dice cuánto espera `waitKey`. Una tecla sustituye
    al tick de esa vuelta; el par de milisegundos que se pierden no se notan.
    """
    estado = start(playlist)
    anterior = reloj()
    try:
        while True:
            paso = playlist[estado.index]
            asset = manifest.letras[paso.label] if paso.label is not None else None
            frame: Any | None = None
            if paso.label is not None and asset is not None:
                imagenes = cuadros[paso.label]
                if asset.es_dinamica and asset.duracion_ms is not None:
                    frame = imagenes[
                        asset_frame(estado, len(imagenes), asset.duracion_ms)
                    ]
                else:
                    frame = imagenes[0]
            ventana.mostrar(componer(Scene(tokens, playlist, estado, asset, frame)))

            tecla = ventana.tecla(config.signs.tick_ms)
            ahora = reloj()
            dt_ms = (ahora - anterior) * 1000.0
            anterior = ahora
            if tecla in _SALIR:
                return estado
            evento = _TECLAS.get(tecla, Tick(dt_ms))
            estado, _ = player_step(estado, evento, playlist, config)
    finally:
        ventana.cerrar()


def reproducir(
    texto: str,
    assets: Path,
    config: Config,
    ventana: Ventana | None = None,
    reloj: Callable[[], float] | None = None,
) -> int:
    """Texto → símbolos → pasos → ventana. Todo lo que puede fallar, falla
    antes de abrir la ventana y con un mensaje concreto."""
    try:
        tokens = text_to_symbols(texto)
    except UnsupportedCharacters as error:
        print(error)
        return 1
    if not any(not isinstance(token, WordGap) for token in tokens):
        print("el texto no tiene ninguna letra que deletrear")
        return 1

    ruta_manifest = assets / MANIFEST_FILENAME
    if not ruta_manifest.is_file():
        print(SIN_MANIFEST.format(ruta=ruta_manifest))
        return 1
    manifest = load_manifest(ruta_manifest)

    try:
        playlist = build_playlist(tokens, manifest, config)
    except ValueError as error:
        print(error)
        return 1

    letras = {paso.label for paso in playlist if paso.label is not None}
    faltan = [
        manifest.letras[label].archivo
        for label in sorted(letras)
        if not (assets / manifest.letras[label].archivo).is_file()
    ]
    if faltan:
        print(
            f"faltan assets en {assets}: {', '.join(faltan)}. Corre lsm-signs render."
        )
        return 1

    if ventana is None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            print(MENSAJE_SIN_EXTRAS.format(modulo=error.name))
            return 1
        ventana = VentanaOpenCV("lsm-signs — texto a señas (deletreo manual)")
    if reloj is None:
        reloj = time.perf_counter

    cuadros = {
        label: load_asset_frames(assets / manifest.letras[label].archivo)
        for label in letras
    }
    print(render_tokens(tokens))
    bucle(
        playlist,
        tokens,
        manifest,
        cuadros,
        config,
        ventana,
        reloj,
        lambda escena: draw_scene(escena, config),
    )
    return 0
