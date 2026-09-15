"""Manifest en disco, candidatas del dataset y render PNG/GIF. Todo en tmp_path."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from lsm.config import Config
from lsm.io.dataset import SampleMetadata, StoredSample, write_sample
from lsm.io.signs import (
    _MARGEN_TEXTO,
    _PANEL_ANCHO,
    MANIFEST_FILENAME,
    _envolver,
    _fuente,
    _layout_panel,
    _ventana_de_simbolos,
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
    Manifest,
    PlayerState,
    Scene,
    SignAsset,
    Step,
    StepKind,
    build_playlist,
    expected_filename,
    project_frames,
    text_to_symbols,
)
from lsm.synthetic import arc_offsets, canonical_hand, moving_sequence, still_sequence
from lsm.types import (
    Distance,
    Handedness,
    LightDirection,
    LightLevel,
    SampleKind,
    Sequence,
)
from lsm.vocabulary import LETTERS, Label

FECHA = datetime(2026, 9, 9, 10, 0, tzinfo=UTC)
HOY = date(2026, 9, 14)


def manifiesto() -> Manifest:
    """Un manifest completo y válido, copiado de `vocabulary.py`. Es el mismo
    helper que en `tests/test_signs.py`; los tests no se importan entre sí."""
    letras: dict[Label, SignAsset] = {}
    for label, letra in LETTERS.items():
        letras[label] = SignAsset(
            letra=letra.display,
            archivo=expected_filename(label),
            es_dinamica=letra.es_dinamica,
            descripcion=letra.descripcion,
            trayectoria=letra.trayectoria,
            pagina=letra.pagina,
            duracion_ms=2000 if letra.es_dinamica else None,
            fuente=AssetSource(
                tipo="esqueleto_desde_dataset",
                muestra=f"s01/2026-09-09-manana/{label}/007.json",
                signer_id="s01",
                lateralidad_original=Handedness.RIGHT,
                espejada=False,
            ),
            revision=AssetReview(fecha=HOY, revisor="tests", resultado="coincide"),
        )
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )


def guardada(label: str, secuencia: Sequence, kind: SampleKind) -> StoredSample:
    metadata = SampleMetadata(
        label=label,
        signer_id="s01",
        session_id="2026-09-09-manana",
        timestamp=FECHA,
        handedness=Handedness.RIGHT,
        light_level=LightLevel.INDOOR,
        light_direction=LightDirection.FRONTAL,
        distance=Distance.MEDIUM,
        mean_luminance=0.5,
        mean_scale_px=90.0,
        kind=kind,
        dispersion=0.01,
        arc_length=0.0 if kind is SampleKind.STATIC else 1.2,
        handedness_swapped=False,
    )
    return StoredSample(metadata=metadata, frames=secuencia.frames)


def test_el_manifest_va_y_vuelve_y_siempre_con_los_mismos_bytes(tmp_path: Path) -> None:
    ruta = tmp_path / MANIFEST_FILENAME
    original = manifiesto()

    save_manifest(original, ruta)
    primera = ruta.read_bytes()
    save_manifest(load_manifest(ruta), ruta)

    assert load_manifest(ruta) == original
    assert ruta.read_bytes() == primera
    assert primera.endswith(b"\n")


def test_las_candidatas_se_agrupan_por_letra_con_su_ruta_relativa(
    tmp_path: Path,
) -> None:
    write_sample(
        tmp_path,
        guardada("A", still_sequence(canonical_hand(), length=8), SampleKind.STATIC),
    )
    write_sample(
        tmp_path,
        guardada("A", still_sequence(canonical_hand(), length=8), SampleKind.STATIC),
    )
    write_sample(
        tmp_path,
        guardada(
            "J", moving_sequence(canonical_hand(), arc_offsets(12)), SampleKind.DYNAMIC
        ),
    )

    candidatas = load_candidates(tmp_path)

    assert sorted(candidatas) == [Label.A, Label.J]
    assert len(candidatas[Label.A]) == 2
    assert candidatas[Label.A][0].path == "s01/2026-09-09-manana/A/001.json"
    assert candidatas[Label.J][0].sample.kind is SampleKind.DYNAMIC


def test_una_muestra_con_otra_version_de_esquema_se_salta(tmp_path: Path) -> None:
    """`read_sample` rechaza otra `schema_version` con `DatasetError`; el
    docstring de `load_candidates` promete saltarla, no reventar (M2)."""
    write_sample(
        tmp_path,
        guardada("A", still_sequence(canonical_hand(), length=8), SampleKind.STATIC),
    )
    mala = tmp_path / "s01" / "2026-09-09-manana" / "A" / "002.json"
    mala.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    candidatas = load_candidates(tmp_path)

    assert len(candidatas[Label.A]) == 1
    assert candidatas[Label.A][0].path == "s01/2026-09-09-manana/A/001.json"


def test_el_png_es_cuadrado_y_del_tamano_pedido(tmp_path: Path) -> None:
    frames = still_sequence(canonical_hand(), length=5).frames
    puntos = project_frames(frames, mirrored=False, canvas_px=200, margin=0.1)
    ruta = tmp_path / "A.png"

    write_static_asset(ruta, puntos, canvas_px=200)

    with Image.open(ruta) as imagen:
        assert imagen.size == (200, 200)
        assert imagen.format == "PNG"


def test_el_gif_lleva_un_frame_por_frame_y_devuelve_su_duracion(tmp_path: Path) -> None:
    frames = moving_sequence(canonical_hand(), arc_offsets(18)).frames
    puntos = project_frames(frames, mirrored=False, canvas_px=160, margin=0.1)
    ruta = tmp_path / "J.gif"

    duracion = write_dynamic_asset(ruta, puntos, canvas_px=160, fps=12)

    assert gif_frame_count(ruta) == 18
    assert duracion == round(1000 * 18 / 12)
    cuadros = load_asset_frames(ruta)
    assert len(cuadros) == 18
    assert cuadros[0].size == (160, 160)
    assert cuadros[0].mode == "RGB"


def test_un_png_se_carga_como_un_solo_cuadro(tmp_path: Path) -> None:
    frames = still_sequence(canonical_hand(), length=2).frames
    puntos = project_frames(frames, mirrored=False, canvas_px=64, margin=0.1)
    ruta = tmp_path / "A.png"
    write_static_asset(ruta, puntos, canvas_px=64)

    assert gif_frame_count(ruta) == 1
    assert len(load_asset_frames(ruta)) == 1


def test_el_cuadro_tiene_asset_a_la_izquierda_y_texto_a_la_derecha() -> None:
    config = Config()
    manifest = manifiesto()
    tokens = text_to_symbols("año")
    playlist = build_playlist(tokens, manifest, config)
    lienzo = Image.new(
        "RGB", (config.signs.canvas_px, config.signs.canvas_px), (255, 0, 0)
    )
    escena = Scene(
        tokens=tokens,
        playlist=playlist,
        state=PlayerState(index=1, elapsed_ms=300.0),
        asset=manifest.letras[Label.ENIE],
        frame=lienzo,
    )

    cuadro = draw_scene(escena, config)

    assert cuadro.width > config.signs.canvas_px
    assert cuadro.height > config.signs.canvas_px
    # El asset se pega tal cual: un píxel del centro del panel izquierdo es rojo.
    centro = config.signs.canvas_px // 2
    assert cuadro.getpixel((centro, centro)) == (255, 0, 0)


def test_una_pausa_se_dibuja_sin_asset() -> None:
    config = Config()
    manifest = manifiesto()
    tokens = text_to_symbols("a b")
    escena = Scene(
        tokens=tokens,
        playlist=build_playlist(tokens, manifest, config),
        state=PlayerState(index=1),
        asset=None,
        frame=None,
    )

    cuadro = draw_scene(escena, config)

    assert cuadro.width > 0


# --------------------------------------------------------------------------- #
# Fuente empaquetada
# --------------------------------------------------------------------------- #


def test_la_fuente_empaquetada_dibuja_ene_con_tilde_y_acentos() -> None:
    """Con Aileron (la fuente por defecto de Pillow) estos glifos no existen y
    salen como el glifo `.notdef` (mismo tamaño e histograma); con DejaVu Sans,
    empaquetada, el glifo real es distinto del `.notdef`."""
    fuente = _fuente(48)
    notdef = fuente.getmask("͸")  # punto de código sin asignar: siempre .notdef

    ene = fuente.getmask("Ñ")
    assert ene.size != notdef.size or ene.histogram() != notdef.histogram()

    e_acento = fuente.getmask("é")
    assert e_acento.size != notdef.size or e_acento.histogram() != notdef.histogram()


# --------------------------------------------------------------------------- #
# La ventana deslizante del pie
# --------------------------------------------------------------------------- #


def test_la_ventana_de_simbolos_contiene_el_actual_y_cabe_en_el_ancho() -> None:
    anchos = [20] * 30

    inicio, fin = _ventana_de_simbolos(anchos, actual=25, disponible=100)

    assert inicio <= 25 < fin
    assert sum(anchos[inicio:fin]) <= 100


def test_la_ventana_de_simbolos_es_completa_si_el_texto_es_corto() -> None:
    anchos = [10, 12, 8, 15]

    inicio, fin = _ventana_de_simbolos(anchos, actual=1, disponible=1000)

    assert (inicio, fin) == (0, len(anchos))


def test_el_bloque_de_estado_no_se_solapa_con_la_descripcion() -> None:
    """K y otras letras con descripción larga empujaban el bloque de estado
    hasta escribirlo encima de la última línea de texto (F1). Se comprueba
    con las 29 letras, estáticas y dinámicas, contra el `canvas_px` por
    defecto: es donde menos aire hay."""
    config = Config()
    manifest = manifiesto()
    medidor = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lienzo = Image.new(
        "RGB", (config.signs.canvas_px, config.signs.canvas_px), (0, 0, 0)
    )
    for label, asset in manifest.letras.items():
        paso = Step(kind=StepKind.LETTER, label=label, duration_ms=1000.0)
        escena = Scene(
            tokens=(label,),
            playlist=(paso,),
            state=PlayerState(index=0, elapsed_ms=250.0),
            asset=asset,
            frame=lienzo,
        )

        cuadro = draw_scene(escena, config)

        lineas = _envolver(
            asset.descripcion, _PANEL_ANCHO - 2 * _MARGEN_TEXTO, _fuente(18), medidor
        )
        y_texto_final, y_barra = _layout_panel(
            asset, config.signs.canvas_px, len(lineas)
        )
        assert y_barra - 28 >= y_texto_final, label
        assert cuadro.height >= int(y_barra) + 40, label


def test_el_pie_no_revienta_con_un_texto_largo_y_el_indice_al_final() -> None:
    config = Config()
    manifest = manifiesto()
    tokens = text_to_symbols("a" * 60)
    playlist = build_playlist(tokens, manifest, config)
    escena = Scene(
        tokens=tokens,
        playlist=playlist,
        state=PlayerState(index=len(playlist) - 1),
        asset=manifest.letras[Label.A],
        frame=None,
    )

    cuadro = draw_scene(escena, config)

    assert cuadro.width > 0
