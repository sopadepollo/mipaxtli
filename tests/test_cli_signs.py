"""Los tres subcomandos de `lsm-signs`, sin cámara ni ventana.

`render` y `verificar` corren de verdad sobre un dataset sintético en tmp_path.
`reproducir` se prueba hasta justo antes de abrir la ventana, y su bucle con una
ventana falsa (Task 8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.cli.signs import bucle, main, reproducir
from lsm.config import Config
from lsm.io.dataset import SampleMetadata, StoredSample, write_sample
from lsm.io.signs import MANIFEST_FILENAME, load_asset_frames, load_manifest
from lsm.signs import build_playlist, manifest_drift, text_to_symbols
from lsm.synthetic import (
    arc_offsets,
    class_hand,
    moving_sequence,
    still_sequence,
    synthetic_samples,
)
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


def test_render_no_escribe_nada_si_una_letra_solo_tiene_grabaciones_del_kind_equivocado(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """J es dinámica; si solo hay grabaciones estáticas de J, `choose_reference`
    la descarta por `kind` y no debe quedar ni un archivo a medio escribir."""
    carpeta_j = dataset / "s01" / "2026-09-09-manana" / "J"
    for ruta in carpeta_j.glob("*.json"):
        ruta.unlink()
    write_sample(
        dataset,
        _stored(
            "J",
            still_sequence(class_hand(1), length=6),
            SampleKind.STATIC,
            "s01",
        ),
    )
    assets = tmp_path / "signs"

    assert render(dataset, assets) == 1

    salida = capsys.readouterr().out
    assert "J" in salida
    assert not (assets / MANIFEST_FILENAME).exists()
    if assets.is_dir():
        assert not list(assets.glob("*.png"))
        assert not list(assets.glob("*.gif"))


def test_render_con_un_manifest_previo_corrupto_lo_dice_y_no_lo_pisa(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    ruta = assets / MANIFEST_FILENAME
    ruta.write_text("{not json", encoding="utf-8")

    assert render(dataset, assets) == 1

    salida = capsys.readouterr().out
    assert MANIFEST_FILENAME in salida
    assert ruta.read_text(encoding="utf-8") == "{not json"


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


# --------------------------------------------------------------------------- #
# reproducir
# --------------------------------------------------------------------------- #


@dataclass
class VentanaFalsa:
    """Devuelve las teclas programadas, luego -1, y cuenta cuadros mostrados."""

    teclas: list[int]
    mostrados: int = 0
    cerrada: bool = False
    esperas: list[int] = field(default_factory=list)

    def mostrar(self, _imagen: object) -> None:
        self.mostrados += 1

    def tecla(self, espera_ms: int) -> int:
        self.esperas.append(espera_ms)
        return self.teclas.pop(0) if self.teclas else -1

    def cerrar(self) -> None:
        self.cerrada = True


class RelojFalso:
    """Avanza `paso_s` en cada lectura."""

    def __init__(self, paso_s: float) -> None:
        self.ahora = 0.0
        self.paso = paso_s

    def __call__(self) -> float:
        self.ahora += self.paso
        return self.ahora


def test_el_bucle_termina_con_q_y_los_ticks_salen_del_reloj(
    dataset: Path, tmp_path: Path
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    config = Config()
    manifest = load_manifest(assets / MANIFEST_FILENAME)
    tokens = text_to_symbols("ab")
    playlist = build_playlist(tokens, manifest, config)
    cuadros = {
        label: load_asset_frames(assets / manifest.letras[label].archivo)
        for label in (Label.A, Label.B)
    }
    # 100 ms por vuelta; 20 vueltas son 2 s: la A (1.5 s) ya pasó a la B.
    ventana = VentanaFalsa(teclas=[-1] * 20 + [ord("q")])

    estado = bucle(
        playlist,
        tokens,
        manifest,
        cuadros,
        config,
        ventana,
        RelojFalso(0.1),
        lambda _escena: None,
    )

    assert estado.index == 1
    assert ventana.mostrados == 21
    assert ventana.cerrada
    assert set(ventana.esperas) == {config.signs.tick_ms}


def test_las_teclas_controlan_el_reproductor(dataset: Path, tmp_path: Path) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    config = Config()
    manifest = load_manifest(assets / MANIFEST_FILENAME)
    tokens = text_to_symbols("abc")
    playlist = build_playlist(tokens, manifest, config)
    cuadros = {
        label: load_asset_frames(assets / manifest.letras[label].archivo)
        for label in (Label.A, Label.B, Label.C)
    }
    ventana = VentanaFalsa(
        teclas=[ord("n"), ord("n"), ord("p"), ord(" "), ord("+"), ord("q")]
    )

    estado = bucle(
        playlist,
        tokens,
        manifest,
        cuadros,
        config,
        ventana,
        RelojFalso(0.001),
        lambda _escena: None,
    )

    assert estado.index == 1
    assert estado.paused
    assert estado.speed == 1.0 + config.signs.speed_step


def test_reproducir_rechaza_caracteres_sin_sena_antes_de_abrir_nada(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = reproducir("hola2", tmp_path, Config())

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "'2'" in salida


def test_reproducir_sin_letras_lo_dice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert reproducir("   ", tmp_path, Config()) == 1
    assert "ninguna letra" in capsys.readouterr().out


def test_reproducir_sin_manifest_apunta_a_render(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert reproducir("casa", tmp_path, Config()) == 1
    assert "lsm-signs render" in capsys.readouterr().out


def test_reproducir_con_un_archivo_que_falta_da_la_ruta(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    (assets / "S.png").unlink()

    codigo = reproducir("casa", assets, Config())

    assert codigo == 1
    assert "S.png" in capsys.readouterr().out


def test_reproducir_con_ventana_inyectada_no_necesita_opencv(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "signs"
    render(dataset, assets)
    ventana = VentanaFalsa(teclas=[ord("q")])

    codigo = reproducir(
        "casa", assets, Config(), ventana=ventana, reloj=RelojFalso(0.01)
    )

    assert codigo == 0
    assert ventana.mostrados == 1
    assert "C A S A" in capsys.readouterr().out
