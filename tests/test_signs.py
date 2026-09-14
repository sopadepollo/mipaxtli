"""La dirección texto → señas, sin disco ni ventana.

`lsm.signs` es código puro por la regla 2 de `CLAUDE.md`: recibe manifests ya
cargados, texto y ticks de reloj, y devuelve estados. Todo lo que hay aquí se
ejercita con datos construidos a mano.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from lsm.config import Config
from lsm.signs import (
    FUENTE_NORMATIVA,
    MANIFEST_SCHEMA_VERSION,
    AssetReview,
    AssetSource,
    Faster,
    Finished,
    Manifest,
    Next,
    PlayerEvent,
    PlayerInput,
    PlayerState,
    Prev,
    Restart,
    SignAsset,
    Slower,
    Step,
    StepKind,
    StepStarted,
    Tick,
    TogglePause,
    UnsupportedCharacters,
    WordGap,
    asset_frame,
    build_playlist,
    expected_filename,
    manifest_drift,
    player_step,
    render_tokens,
    start,
    text_to_symbols,
)
from lsm.types import Handedness
from lsm.vocabulary import LETTERS, Label, spec

HOY = date(2026, 9, 14)


def asset(label: Label, **cambios: object) -> SignAsset:
    """Una entrada válida del manifest, copiada de `vocabulary.py`."""
    letra = spec(label)
    base: dict[str, object] = {
        "letra": letra.display,
        "archivo": expected_filename(label),
        "es_dinamica": letra.es_dinamica,
        "descripcion": letra.descripcion,
        "trayectoria": letra.trayectoria,
        "pagina": letra.pagina,
        "duracion_ms": 2000 if letra.es_dinamica else None,
        "fuente": AssetSource(
            tipo="esqueleto_desde_dataset",
            muestra=f"s01/2026-09-09-manana/{label}/007.json",
            signer_id="s01",
            lateralidad_original=Handedness.RIGHT,
            espejada=False,
        ),
        "revision": AssetReview(fecha=HOY, revisor="tests", resultado="coincide"),
    }
    base.update(cambios)
    return SignAsset.model_validate(base)


def manifiesto(**cambios: SignAsset) -> Manifest:
    letras: dict[Label, SignAsset] = {label: asset(label) for label in LETTERS}
    for key, value in cambios.items():
        letras[Label(key)] = value
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )


# --------------------------------------------------------------------------- #
# Esquema
# --------------------------------------------------------------------------- #


def test_el_nombre_del_archivo_lo_fija_la_etiqueta() -> None:
    assert expected_filename(Label.A) == "A.png"
    assert expected_filename(Label.J) == "J.gif"
    assert expected_filename(Label.DOBLE_L) == "DOBLE_L.gif"


def test_un_manifest_completo_no_deriva_del_glosario() -> None:
    assert manifest_drift(manifiesto()) == []


def test_una_dinamica_con_imagen_fija_se_rechaza() -> None:
    """Las dinámicas necesitan movimiento: GIF, no PNG. Es la regla del spec
    hecha tipo, y falla al cargar en vez de al reproducir."""
    with pytest.raises(ValidationError, match="gif"):
        asset(Label.J, archivo="J.png")


def test_una_estatica_con_gif_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="png"):
        asset(Label.A, archivo="A.gif")


def test_una_dinamica_sin_duracion_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="duracion_ms"):
        asset(Label.J, duracion_ms=None)


def test_una_estatica_con_duracion_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="duracion_ms"):
        asset(Label.A, duracion_ms=1500)


def test_la_muestra_de_origen_tiene_la_forma_del_dataset() -> None:
    with pytest.raises(ValidationError, match="firmante/sesion"):
        AssetSource(
            tipo="esqueleto_desde_dataset",
            muestra="J.json",
            signer_id="s01",
            lateralidad_original=Handedness.RIGHT,
            espejada=False,
        )


def test_el_archivo_tiene_que_llamarse_como_la_etiqueta() -> None:
    with pytest.raises(ValidationError, match=r"A\.png"):
        manifiesto(A=asset(Label.A, archivo="B.png"))


def test_la_muestra_de_origen_tiene_que_ser_de_la_misma_letra() -> None:
    fuente = AssetSource(
        tipo="esqueleto_desde_dataset",
        muestra="s01/2026-09-09-manana/B/001.json",
        signer_id="s01",
        lateralidad_original=Handedness.RIGHT,
        espejada=False,
    )
    with pytest.raises(ValidationError, match="muestra"):
        manifiesto(A=asset(Label.A, fuente=fuente))


def test_la_clase_negativa_no_es_una_letra() -> None:
    letras = {label: asset(label) for label in LETTERS}
    letras[Label.NONE] = asset(Label.A)
    with pytest.raises(ValidationError, match="NONE"):
        Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            fuente_normativa=FUENTE_NORMATIVA,
            letras=letras,
        )


def test_otra_version_de_esquema_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION + 1,
            fuente_normativa=FUENTE_NORMATIVA,
            letras={label: asset(label) for label in LETTERS},
        )


def test_campos_desconocidos_se_rechazan() -> None:
    with pytest.raises(ValidationError):
        AssetReview.model_validate(
            {"fecha": HOY, "revisor": "x", "resultado": "coincide", "extra": 1}
        )


# --------------------------------------------------------------------------- #
# Deriva contra vocabulary.py
# --------------------------------------------------------------------------- #


def test_una_letra_que_falta_es_deriva() -> None:
    letras = {label: asset(label) for label in LETTERS}
    del letras[Label.Q]
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )

    deriva = manifest_drift(manifest)

    assert len(deriva) == 1
    assert "Q" in deriva[0]
    assert "falta" in deriva[0]


def test_una_descripcion_distinta_es_deriva() -> None:
    deriva = manifest_drift(manifiesto(M=asset(Label.M, descripcion="otra cosa")))

    assert len(deriva) == 1
    assert "M" in deriva[0]
    assert "descripcion" in deriva[0]


def test_una_pagina_distinta_es_deriva() -> None:
    deriva = manifest_drift(manifiesto(Z=asset(Label.Z, pagina=1)))

    assert deriva
    assert "pagina" in deriva[0]


def test_la_forma_de_escribir_la_letra_tambien_se_compara() -> None:
    deriva = manifest_drift(manifiesto(ENIE=asset(Label.ENIE, letra="N")))

    assert deriva
    assert "letra" in deriva[0]


# --------------------------------------------------------------------------- #
# Texto → símbolos
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("llave", (Label.DOBLE_L, Label.A, Label.V, Label.E)),
        ("carro", (Label.C, Label.A, Label.DOBLE_R, Label.O)),
        ("año", (Label.A, Label.ENIE, Label.O)),
        ("chico", (Label.C, Label.H, Label.I, Label.C, Label.O)),
        ("ll", (Label.DOBLE_L,)),
        ("lll", (Label.DOBLE_L, Label.L)),
        ("Ñ", (Label.ENIE,)),
        (
            "pingüino",
            (Label.P, Label.I, Label.N, Label.G, Label.U, Label.I, Label.N, Label.O),
        ),
    ],
)
def test_texto_a_simbolos(texto: str, esperado: tuple[Label, ...]) -> None:
    assert text_to_symbols(texto) == esperado


def test_los_espacios_separan_palabras_y_no_se_acumulan() -> None:
    tokens = text_to_symbols("  Árbol  verde ")

    assert tokens == (
        Label.A, Label.R, Label.B, Label.O, Label.L,
        WordGap(),
        Label.V, Label.E, Label.R, Label.D, Label.E,
    )  # fmt: skip


def test_la_enie_sobrevive_a_quitar_los_acentos() -> None:
    """En NFD la ñ es `n` + tilde: quitar marcas combinantes sin cuidado la
    convertiría en N y "año" se deletrearía como "ano"."""
    assert text_to_symbols("ñandú") == (Label.ENIE, Label.A, Label.N, Label.D, Label.U)


def test_un_caracter_sin_sena_se_rechaza_con_la_lista_exacta() -> None:
    with pytest.raises(UnsupportedCharacters) as excinfo:
        text_to_symbols("hola2! 2")

    assert excinfo.value.chars == ("2", "!")


def test_un_texto_sin_letras_da_una_tupla_vacia() -> None:
    assert text_to_symbols("   ") == ()
    assert text_to_symbols("") == ()


def test_los_simbolos_se_muestran_como_se_escriben() -> None:
    tokens = text_to_symbols("año ll")

    assert render_tokens(tokens) == "A Ñ O · LL"
    # Also check DOBLE_R
    assert render_tokens(text_to_symbols("carro")) == "C A RR O"


def test_un_caracter_que_se_expande_al_mayusculas_se_rechaza() -> None:
    """ß (German sharp s) expands to SS when uppercased, causing index mismatch."""
    with pytest.raises(UnsupportedCharacters) as excinfo:
        text_to_symbols("ß")

    assert excinfo.value.chars == ("ß",)


def test_un_caracter_expandible_con_otros_invalidos_se_reportan_juntos() -> None:
    with pytest.raises(UnsupportedCharacters) as excinfo:
        text_to_symbols("ß2")

    assert excinfo.value.chars == ("ß", "2")


# --------------------------------------------------------------------------- #
# Lista de pasos
# --------------------------------------------------------------------------- #

CONFIG = Config()


def test_las_duraciones_salen_de_la_configuracion_y_del_manifest() -> None:
    pasos = build_playlist(text_to_symbols("aj a"), manifiesto(), CONFIG)

    assert [paso.kind for paso in pasos] == [
        StepKind.LETTER, StepKind.LETTER, StepKind.GAP, StepKind.LETTER
    ]  # fmt: skip
    assert pasos[0].duration_ms == CONFIG.signs.static_hold_ms
    assert pasos[1].duration_ms == 2000 * CONFIG.signs.dynamic_loops
    assert pasos[2].duration_ms == CONFIG.signs.word_gap_ms
    assert pasos[2].label is None


def test_una_letra_sin_asset_no_entra_en_la_lista_en_silencio() -> None:
    letras = {label: asset(label) for label in LETTERS}
    del letras[Label.J]
    incompleto = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fuente_normativa=FUENTE_NORMATIVA,
        letras=letras,
    )
    with pytest.raises(ValueError, match="J"):
        build_playlist((Label.J,), incompleto, CONFIG)


def test_una_lista_vacia_no_arranca() -> None:
    with pytest.raises(ValueError, match="vac"):
        start(())


# --------------------------------------------------------------------------- #
# Reproductor
# --------------------------------------------------------------------------- #

DOS_PASOS = (
    Step(kind=StepKind.LETTER, label=Label.A, duration_ms=1000.0),
    Step(kind=StepKind.LETTER, label=Label.B, duration_ms=500.0),
)


def avanzar(
    estado: PlayerState, *eventos: PlayerInput
) -> tuple[PlayerState, list[PlayerEvent]]:
    emitidos: list[PlayerEvent] = []
    for evento in eventos:
        estado, nuevos = player_step(estado, evento, DOS_PASOS, CONFIG)
        emitidos.extend(nuevos)
    return estado, emitidos


def test_los_ticks_acumulan_y_al_agotar_el_paso_pasan_al_siguiente() -> None:
    estado, eventos = avanzar(start(DOS_PASOS), Tick(400.0), Tick(400.0))
    assert estado.index == 0
    assert estado.elapsed_ms == 800.0
    assert eventos == []

    estado, eventos = avanzar(estado, Tick(200.0))
    assert estado.index == 1
    assert estado.elapsed_ms == 0.0
    assert eventos == [StepStarted(index=1)]


def test_al_agotar_el_ultimo_paso_termina_y_se_queda_en_el() -> None:
    estado = PlayerState(index=1, elapsed_ms=400.0)

    estado, eventos = avanzar(estado, Tick(100.0))
    assert estado.finished
    assert estado.index == 1
    assert eventos == [Finished()]

    otra, mas = avanzar(estado, Tick(5000.0))
    assert otra == estado
    assert mas == []


def test_en_pausa_los_ticks_no_avanzan() -> None:
    estado, _ = avanzar(start(DOS_PASOS), TogglePause(), Tick(5000.0))
    assert estado.paused
    assert estado.elapsed_ms == 0.0

    estado, _ = avanzar(estado, TogglePause(), Tick(100.0))
    assert not estado.paused
    assert estado.elapsed_ms == 100.0


def test_la_velocidad_multiplica_el_tiempo() -> None:
    estado, _ = avanzar(start(DOS_PASOS), Faster(), Tick(100.0))

    assert estado.speed == 1.0 + CONFIG.signs.speed_step
    assert estado.elapsed_ms == pytest.approx(100.0 * estado.speed)


def test_la_velocidad_queda_acotada() -> None:
    muchas = [Faster()] * 100
    estado, _ = avanzar(start(DOS_PASOS), *muchas)
    assert estado.speed == CONFIG.signs.speed_max

    pocas = [Slower()] * 100
    estado, _ = avanzar(estado, *pocas)
    assert estado.speed == CONFIG.signs.speed_min


def test_siguiente_y_anterior_quedan_acotados() -> None:
    estado, eventos = avanzar(start(DOS_PASOS), Tick(300.0), Next())
    assert estado.index == 1
    assert estado.elapsed_ms == 0.0
    assert eventos == [StepStarted(index=1)]

    estado, eventos = avanzar(estado, Next())
    assert estado.finished
    assert estado.index == 1
    assert eventos == [Finished()]

    estado, eventos = avanzar(estado, Prev())
    assert not estado.finished
    assert estado.index == 0
    assert eventos == [StepStarted(index=0)]

    estado, eventos = avanzar(estado, Tick(300.0), Prev())
    assert estado.index == 0
    assert estado.elapsed_ms == 0.0


def test_reiniciar_conserva_la_velocidad() -> None:
    estado, _ = avanzar(start(DOS_PASOS), Faster(), Next(), Next())
    assert estado.finished

    estado, eventos = avanzar(estado, Restart())
    assert estado == PlayerState(speed=1.0 + CONFIG.signs.speed_step)
    assert eventos == [StepStarted(index=0)]


def test_el_frame_del_gif_da_vueltas_con_los_loops() -> None:
    n_frames, duracion = 24, 2000
    assert asset_frame(PlayerState(elapsed_ms=0.0), n_frames, duracion) == 0
    assert asset_frame(PlayerState(elapsed_ms=1000.0), n_frames, duracion) == 12
    assert asset_frame(PlayerState(elapsed_ms=2000.0), n_frames, duracion) == 0
    assert asset_frame(PlayerState(elapsed_ms=3999.0), n_frames, duracion) == 23
