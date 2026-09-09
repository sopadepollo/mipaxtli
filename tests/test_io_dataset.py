"""El dataset en disco: que lo escrito sea exactamente lo grabado.

Todo lo que este archivo comprueba se reduce a una promesa: **si cambia la
normalización, el dataset se re-deriva con un comando en vez de regrabarse con
personas frente a la cámara**. Esa promesa se apoya en guardar landmarks crudos y
en que el viaje a JSON y de vuelta no pierda ni un bit; si cualquiera de las dos
cosas falla, la Fase 1 entera queda en falso.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.io.dataset import (
    CONSENT_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
    Consent,
    DatasetError,
    SampleMetadata,
    StoredSample,
    check_identifier,
    consent_path,
    count_samples,
    iter_sample_paths,
    iter_samples,
    label_dir,
    load_consents,
    may_store_video,
    next_sample_index,
    now,
    read_sample,
    sample_path,
    save_consent,
    write_sample,
)
from lsm.synthetic import arc_offsets, canonical_hand, moving_sequence, still_sequence
from lsm.types import (
    Distance,
    FrameStream,
    Handedness,
    InvalidFrame,
    InvalidReason,
    LightDirection,
    LightLevel,
    RawFrame,
    SampleKind,
)

FECHA = datetime(2026, 9, 8, 11, 30, tzinfo=UTC)


def metadatos(**cambios: object) -> SampleMetadata:
    base: dict[str, object] = {
        "label": "A",
        "signer_id": "s01",
        "session_id": "2026-09-08-manana",
        "timestamp": FECHA,
        "handedness": Handedness.RIGHT,
        "light_level": LightLevel.INDOOR,
        "light_direction": LightDirection.FRONTAL,
        "distance": Distance.MEDIUM,
        "mean_luminance": 0.44,
        "mean_scale_px": 98.5,
        "kind": SampleKind.STATIC,
        "dispersion": 0.0123,
        "arc_length": 0.0,
        "handedness_swapped": True,
    }
    base.update(cambios)
    return SampleMetadata(**base)  # type: ignore[arg-type]


def muestra(frames: FrameStream | None = None, **cambios: object) -> StoredSample:
    if frames is None:
        frames = still_sequence(canonical_hand(), length=24).frames
    return StoredSample(metadata=metadatos(**cambios), frames=frames)


# --------------------------------------------------------------------------- #
# Ida y vuelta por disco
# --------------------------------------------------------------------------- #


def test_una_muestra_sobrevive_el_viaje_a_disco(tmp_path: Path) -> None:
    original = muestra()

    ruta = write_sample(tmp_path, original)
    recuperada = read_sample(ruta)

    assert recuperada.metadata == original.metadata
    assert recuperada.frames == original.frames


def test_los_landmarks_no_pierden_un_solo_bit(tmp_path: Path) -> None:
    """La promesa entera del dataset cuelga de esto.

    Si el viaje a JSON redondeara, las features re-derivadas no serían las del
    momento de grabar y "cambiar la normalización cuesta un comando" sería falso
    — pero nadie se enteraría hasta la Fase 2, mirando métricas raras.
    """
    original = muestra(frames=moving_sequence(canonical_hand(), arc_offsets(9)).frames)

    recuperada = read_sample(write_sample(tmp_path, original))

    for antes, despues in zip(original.frames, recuperada.frames, strict=True):
        assert isinstance(antes, RawFrame)
        assert isinstance(despues, RawFrame)
        for a, d in zip(antes.landmarks, despues.landmarks, strict=True):
            assert (d.x, d.y, d.z) == (a.x, a.y, a.z)


def test_los_huecos_se_guardan_como_huecos(tmp_path: Path) -> None:
    """`docs/dataset-schema.md`: si el hueco se perdiera al guardar, una secuencia
    interrumpida se convertiría en una continua y el archivo mentiría sobre lo que
    ocurrió frente a la cámara."""
    frames = still_sequence(canonical_hand(), length=6).frames
    con_hueco: FrameStream = (
        *frames[:3],
        InvalidFrame(reason=InvalidReason.NO_HAND, detail="salió del encuadre"),
        *frames[3:],
    )

    recuperada = read_sample(write_sample(tmp_path, muestra(frames=con_hueco)))

    hueco = recuperada.frames[3]
    assert isinstance(hueco, InvalidFrame)
    assert hueco.reason is InvalidReason.NO_HAND
    assert hueco.detail == "salió del encuadre"


def test_una_muestra_con_huecos_no_se_puede_entrenar() -> None:
    """Coser dos trozos inventaría un movimiento que nadie ejecutó, y quedarse con
    el más largo entregaría una seña recortada con la etiqueta de la completa. Las
    dos cosas envenenan el entrenamiento en silencio."""
    frames = still_sequence(canonical_hand(), length=6).frames
    con_hueco: FrameStream = (
        *frames[:3],
        InvalidFrame(reason=InvalidReason.NO_HAND),
        *frames[3:],
    )

    with pytest.raises(DatasetError, match="huecos"):
        muestra(frames=con_hueco).to_sample()


def test_una_muestra_intacta_si_se_convierte_en_muestra_entrenable() -> None:
    convertida = muestra().to_sample()

    assert convertida.label == "A"
    assert convertida.signer_id == "s01"
    assert convertida.mean_scale_px == 98.5
    assert len(convertida.sequence) == 24


def test_un_archivo_de_otra_version_se_rechaza(tmp_path: Path) -> None:
    ruta = write_sample(tmp_path, muestra())
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["schema_version"] = SAMPLE_SCHEMA_VERSION + 1
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="schema_version"):
        read_sample(ruta)


def test_un_timestamp_sin_zona_horaria_se_rechaza(tmp_path: Path) -> None:
    """Sin offset, dos sesiones grabadas en husos distintos no se pueden ordenar,
    y ordenar sesiones es lo primero que se hace al depurar por qué una tanda salió
    peor que otra."""
    ruta = write_sample(tmp_path, muestra())
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["timestamp"] = "2026-09-08T11:30:00"
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="zona horaria"):
        read_sample(ruta)


def test_now_siempre_lleva_zona_horaria() -> None:
    assert now().tzinfo is not None


# --------------------------------------------------------------------------- #
# Disposición en disco
# --------------------------------------------------------------------------- #


def test_la_jerarquia_empieza_por_la_persona(tmp_path: Path) -> None:
    """El split es leave-one-signer-out y el borrado a petición es por persona:
    las dos operaciones que más se hacen sobre este árbol son por firmante."""
    directorio = label_dir(tmp_path, "s01", "sesion-a", "ENIE")

    assert directorio == tmp_path / "s01" / "sesion-a" / "ENIE"


def test_los_numeros_de_muestra_ordenan_bien(tmp_path: Path) -> None:
    assert sample_path(tmp_path, 7).name == "007.json"
    assert sample_path(tmp_path, 123).name == "123.json"


def test_borrar_una_muestra_mala_no_reutiliza_su_numero(tmp_path: Path) -> None:
    """Si la 003 se borró por mala, la siguiente es la 006. Reutilizar el 003
    haría que la bitácora de la sesión hablara de una muestra distinta a la que
    hay en disco."""
    for _ in range(5):
        write_sample(tmp_path, muestra())
    directorio = label_dir(tmp_path, "s01", "2026-09-08-manana", "A")
    (directorio / "003.json").unlink()

    assert next_sample_index(directorio) == 6


def test_el_indice_de_una_carpeta_que_no_existe_es_uno(tmp_path: Path) -> None:
    assert next_sample_index(tmp_path / "todavia-no") == 1


def test_contar_muestras_no_abre_los_archivos(tmp_path: Path) -> None:
    """Alimenta el contador del preview, que se refresca treinta veces por segundo
    y no puede permitirse releer el dataset entero."""
    for _ in range(3):
        write_sample(tmp_path, muestra())
    for _ in range(2):
        write_sample(tmp_path, muestra(label="B"))

    assert count_samples(tmp_path, "s01", "2026-09-08-manana") == {"A": 3, "B": 2}


def test_contar_una_sesion_que_no_existe_da_vacio(tmp_path: Path) -> None:
    assert count_samples(tmp_path, "s01", "no-existe") == {}


def test_el_recorrido_del_dataset_es_estable(tmp_path: Path) -> None:
    """Un orden que dependa del sistema de archivos haría que dos ejecuciones de
    la evaluación no fueran comparables entre sí."""
    for etiqueta in ("B", "A", "ENIE"):
        write_sample(tmp_path, muestra(label=etiqueta))

    rutas = list(iter_sample_paths(tmp_path))

    assert rutas == sorted(rutas)
    assert [ruta.parent.name for ruta in rutas] == ["A", "B", "ENIE"]
    assert len(list(iter_samples(tmp_path))) == 3


def test_recorrer_un_dataset_inexistente_no_falla(tmp_path: Path) -> None:
    assert list(iter_sample_paths(tmp_path / "nada")) == []


def test_el_registro_de_consentimiento_no_se_confunde_con_una_muestra(
    tmp_path: Path,
) -> None:
    """Vive en la raíz y no en `<firmante>/<sesion>/<letra>/`, así que el patrón
    del recorrido no lo alcanza. Si lo alcanzara, `read_sample` reventaría al leer
    un archivo que no es una muestra."""
    write_sample(tmp_path, muestra())
    save_consent(tmp_path, Consent(signer_id="s01", video=False, fecha=FECHA))

    assert consent_path(tmp_path).is_file()
    assert len(list(iter_sample_paths(tmp_path))) == 1


# --------------------------------------------------------------------------- #
# Identificadores
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("valido", ["s01", "S-01", "sesion_2026-09-08", "a", "0"])
def test_los_identificadores_razonables_se_admiten(valido: str) -> None:
    assert check_identifier("signer_id", valido) == valido


@pytest.mark.parametrize(
    "invalido",
    [
        "",
        "..",
        "../otro",
        "s01/x",
        "s01\\x",
        "-empieza-con-guion",
        "con espacio",
        "ñ",
        "x" * 65,
    ],
)
def test_un_identificador_peligroso_se_rechaza(invalido: str) -> None:
    """Estos valores viajan a nombres de carpeta. Un `../` en un `signer_id`
    escribiría fuera de `data/raw/`, y una `ñ` o un espacio sobreviven en Linux
    pero no en todas partes."""
    with pytest.raises(DatasetError, match="inválido"):
        check_identifier("signer_id", invalido)


def test_una_muestra_no_puede_escapar_de_la_raiz(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="inválido"):
        write_sample(tmp_path, muestra(signer_id="../fuera"))


# --------------------------------------------------------------------------- #
# Consentimiento
# --------------------------------------------------------------------------- #


def test_sin_registro_no_hay_permiso_para_guardar_video(tmp_path: Path) -> None:
    """El valor por defecto de una pregunta sobre permisos es "no": un archivo que
    falta es una autorización que no se pidió, nunca una que se dio."""
    assert load_consents(tmp_path) == {}
    assert may_store_video(tmp_path, "s01") is False


def test_registrarse_sin_autorizar_video_tampoco_da_permiso(tmp_path: Path) -> None:
    """Los landmarks no identifican a nadie; el video sí. Aparecer en el registro
    no es lo mismo que haber autorizado que se guarden cuadros."""
    save_consent(tmp_path, Consent(signer_id="s01", video=False, fecha=FECHA))

    assert may_store_video(tmp_path, "s01") is False


def test_el_consentimiento_de_video_se_registra_y_se_relee(tmp_path: Path) -> None:
    save_consent(
        tmp_path,
        Consent(signer_id="s01", video=True, fecha=FECHA, referencia="folio 3"),
    )

    registro = load_consents(tmp_path)

    assert may_store_video(tmp_path, "s01") is True
    assert registro["s01"].referencia == "folio 3"
    assert registro["s01"].fecha == FECHA


def test_el_consentimiento_de_una_persona_no_alcanza_a_otra(tmp_path: Path) -> None:
    save_consent(tmp_path, Consent(signer_id="s01", video=True, fecha=FECHA))

    assert may_store_video(tmp_path, "s01") is True
    assert may_store_video(tmp_path, "s02") is False


def test_registrar_a_alguien_no_borra_a_los_demas(tmp_path: Path) -> None:
    save_consent(tmp_path, Consent(signer_id="s01", video=True, fecha=FECHA))
    save_consent(tmp_path, Consent(signer_id="s02", video=False, fecha=FECHA))

    assert set(load_consents(tmp_path)) == {"s01", "s02"}
    assert may_store_video(tmp_path, "s01") is True


def test_se_puede_revocar_el_permiso_de_video(tmp_path: Path) -> None:
    """Quien firma puede cambiar de opinión, y el registro tiene que poder
    reflejarlo (`ARQUITECTURA.md` §4.11)."""
    save_consent(tmp_path, Consent(signer_id="s01", video=True, fecha=FECHA))
    save_consent(tmp_path, Consent(signer_id="s01", video=False, fecha=FECHA))

    assert may_store_video(tmp_path, "s01") is False


def test_un_registro_de_otra_version_se_rechaza(tmp_path: Path) -> None:
    save_consent(tmp_path, Consent(signer_id="s01", video=True, fecha=FECHA))
    ruta = consent_path(tmp_path)
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["schema_version"] = CONSENT_SCHEMA_VERSION + 1
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="schema_version"):
        load_consents(tmp_path)
