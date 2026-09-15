"""Los tres subcomandos de `lsm-signs`, sin cámara ni ventana.

`render` y `verificar` corren de verdad sobre un dataset sintético en tmp_path.
`reproducir` se prueba hasta justo antes de abrir la ventana, y su bucle con una
ventana falsa (Task 8).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.cli.signs import main
from lsm.io.dataset import SampleMetadata, StoredSample, write_sample
from lsm.io.signs import MANIFEST_FILENAME, load_manifest
from lsm.signs import manifest_drift
from lsm.synthetic import arc_offsets, class_hand, moving_sequence, synthetic_samples
from lsm.types import (
    Distance,
    Handedness,
    LightDirection,
    LightLevel,
    SampleKind,
    Sequence,
)
from lsm.vocabulary import ALPHABET, DYNAMIC_LABELS, LETTERS, Label

RAIZ_REPO = Path(__file__).resolve().parent.parent
CONFIG_YAML = RAIZ_REPO / "config.yaml"
FECHA = datetime(2026, 9, 9, 10, 0, tzinfo=UTC)


def _stored(
    label: str, secuencia: Sequence, kind: SampleKind, signer: str
) -> StoredSample:
    metadata = SampleMetadata(
        label=label,
        signer_id=signer,
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


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Un data/raw con las 29 letras: estáticas de `synthetic_samples`,
    dinámicas como un trazo en gancho sobre una mano distinta por letra."""
    raiz = tmp_path / "raw"
    estaticas = tuple(
        sorted(str(label) for label in ALPHABET if label not in DYNAMIC_LABELS)
    )
    for muestra in synthetic_samples(
        estaticas, signers=1, sessions=1, repetitions=2, frames=6
    ):
        write_sample(
            raiz, _stored(muestra.label, muestra.sequence, SampleKind.STATIC, "s01")
        )
    for ordinal, label in enumerate(sorted(DYNAMIC_LABELS)):
        trazo = moving_sequence(class_hand(ordinal + 1), arc_offsets(12))
        write_sample(raiz, _stored(str(label), trazo, SampleKind.DYNAMIC, "s01"))
    return raiz


def render(dataset: Path, assets: Path) -> int:
    return main(
        [
            "render",
            "--raw", str(dataset),
            "--assets", str(assets),
            "--config", str(CONFIG_YAML),
            "--revisor", "tests",
            "--hoy", "2026-09-14",
        ]
    )  # fmt: skip


def test_render_produce_29_assets_y_un_manifest_sin_deriva(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"

    assert render(dataset, assets) == 0

    manifest = load_manifest(assets / MANIFEST_FILENAME)
    assert set(manifest.letras) == set(LETTERS)
    assert manifest_drift(manifest) == []
    for label, entrada in manifest.letras.items():
        assert (assets / entrada.archivo).is_file(), label
        assert entrada.revision.resultado == "pendiente"
        assert entrada.revision.revisor == "tests"
        assert entrada.fuente.muestra.startswith("s01/2026-09-09-manana/")
    assert "29" in capsys.readouterr().out


def test_render_es_determinista(dataset: Path, tmp_path: Path) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    primera = (assets / MANIFEST_FILENAME).read_bytes()

    render(dataset, assets)

    assert (assets / MANIFEST_FILENAME).read_bytes() == primera


def test_un_re_render_conserva_la_revision_si_la_muestra_no_cambio(
    dataset: Path, tmp_path: Path
) -> None:
    """Regenerar los assets no puede borrar trabajo humano sin avisar."""
    assets = tmp_path / "signs"
    render(dataset, assets)
    ruta = assets / MANIFEST_FILENAME
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["letras"]["A"]["revision"] = {
        "fecha": "2026-09-13",
        "revisor": "persona",
        "resultado": "coincide",
        "nota": "pulgar visible",
    }
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    render(dataset, assets)

    manifest = load_manifest(ruta)
    assert manifest.letras[Label.A].revision.resultado == "coincide"
    assert manifest.letras[Label.A].revision.nota == "pulgar visible"
    assert manifest.letras[Label.B].revision.resultado == "pendiente"


def test_render_sin_una_letra_no_escribe_nada_y_dice_cual(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for ruta in (dataset / "s01" / "2026-09-09-manana" / "Q").glob("*.json"):
        ruta.unlink()
    assets = tmp_path / "signs"

    assert render(dataset, assets) == 1

    assert not (assets / MANIFEST_FILENAME).exists()
    assert "Q" in capsys.readouterr().out


def test_verificar_exige_archivos_presentes_y_revisiones_cerradas(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    (assets / "J.gif").unlink()

    codigo = main(["verificar", "--assets", str(assets), "--config", str(CONFIG_YAML)])

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "J.gif" in salida
    assert "pendiente" in salida


def test_verificar_sin_manifest_lo_dice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = main(
        ["verificar", "--assets", str(tmp_path), "--config", str(CONFIG_YAML)]
    )

    assert codigo == 1
    assert "lsm-signs render" in capsys.readouterr().out
