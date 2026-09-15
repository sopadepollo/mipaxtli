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
from datetime import date
from pathlib import Path
from typing import Final

from lsm.config import Config, load_config
from lsm.io.signs import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_RAW_DIR,
    MANIFEST_FILENAME,
    gif_frame_count,
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
    Manifest,
    SignAsset,
    choose_reference,
    expected_filename,
    manifest_drift,
    project_frames,
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
    nada si a alguna letra le faltan candidatas."""
    candidatas = load_candidates(raw)
    faltan = [str(label) for label in ALPHABET if not candidatas.get(label)]
    if faltan:
        print(f"sin muestras en {raw} para: {', '.join(faltan)}. No se escribe nada.")
        return 1

    ruta_manifest = assets / MANIFEST_FILENAME
    previo = load_manifest(ruta_manifest) if ruta_manifest.is_file() else None

    letras: dict[Label, SignAsset] = {}
    for label in ALPHABET:
        letra = LETTERS[label]
        eleccion = choose_reference(candidatas[label], label, config)
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


def reproducir(texto: str, assets: Path, config: Config) -> int:
    raise NotImplementedError("Task 8")
