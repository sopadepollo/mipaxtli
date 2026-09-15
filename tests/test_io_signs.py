"""Manifest en disco, candidatas del dataset y render PNG/GIF. Todo en tmp_path."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from PIL import Image

from lsm.io.dataset import SampleMetadata, StoredSample, write_sample
from lsm.io.signs import (
    MANIFEST_FILENAME,
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
    SignAsset,
    expected_filename,
    project_frames,
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
