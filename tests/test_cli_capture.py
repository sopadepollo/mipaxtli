"""El criterio de aceptación de la Fase 1, sin webcam.

> Grabar 20 muestras de una letra, verificar que se re-derivan features idénticas
> a las del pipeline, y que la suite sigue pasando sin cámara.

Las veinte muestras se graban por el mismo camino que usa la sesión interactiva:
`guardar_muestra` es la única puerta por la que una muestra llega al dataset, y lo
que este archivo ejercita es esa puerta, no una reimplementación de al lado. Lo
único que falta respecto de una sesión real es la cámara, el teclado y el dibujo
— y esas tres cosas no deciden nada sobre lo que se guarda.

"Idénticas" significa **igualdad exacta**, no dentro de una tolerancia. La
tolerancia de `1e-6` del contrato es para la reimplementación en TypeScript de la
Fase 7; aquí es el mismo código sobre los mismos números, y cualquier diferencia
delata que el viaje a disco perdió precisión.

Aquí, y solo aquí. Este archivo escribe y relee dentro del mismo proceso y la
misma máquina, que es lo que vuelve legítima la igualdad exacta. `lsm-capture
verificar` corre sobre un dataset grabado en el host y verificado en WSL, con dos
libm distintas detrás de `atan2`, y por eso usa `SIGMA_REL_TOL`. Los dos
criterios conviven a propósito: ver
`docs/adr/0009-verificacion-entre-plataformas.md`.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lsm.capture import BufferedFrame, Rejection
from lsm.cli.capture import (
    Condiciones,
    Guardada,
    _modo_por_defecto,
    _resolver_letras,
    _resolver_video,
    guardar_muestra,
    main,
)
from lsm.config import Config
from lsm.features import ExtractionRejected, extract_sequence_features
from lsm.io.calibration import Calibration, camera_key, save_calibration
from lsm.io.dataset import (
    Consent,
    iter_samples,
    may_store_video,
    read_sample,
    save_consent,
    trial_root,
)
from lsm.synthetic import (
    arc_offsets,
    canonical_hand,
    moving_sequence,
    still_sequence,
    synthetic_samples,
    translated,
)
from lsm.types import (
    HANDEDNESS_CONVENTION,
    Distance,
    InvalidFrame,
    InvalidReason,
    LightDirection,
    LightLevel,
    SampleKind,
)
from lsm.vocabulary import ALPHABET, Label

CONDICIONES = Condiciones(
    light_level=LightLevel.INDOOR,
    light_direction=LightDirection.FRONTAL,
    distance=Distance.MEDIUM,
)


def repeticion(config: Config, indice: int) -> tuple[BufferedFrame, ...]:
    """Una repetición de una letra estática, ligeramente distinta a las demás.

    Se desplaza unos píxeles por repetición porque veinte muestras idénticas no
    prueban gran cosa: en un dataset real cada repetición cae en otro sitio del
    encuadre, y conviene que las veinte de este test tampoco sean el mismo
    archivo veinte veces.
    """
    mano = translated(canonical_hand(), dx=3.0 * indice, dy=-2.0 * indice)
    secuencia = still_sequence(mano, length=config.capture.static_frames)
    return tuple(
        BufferedFrame(slot=frame, luminance=0.40 + 0.001 * indice)
        for frame in secuencia.frames
    )


def grabar_veinte(raiz: Path, config: Config) -> list[Guardada]:
    guardadas: list[Guardada] = []
    for indice in range(20):
        resultado = guardar_muestra(
            frames=repeticion(config, indice),
            kind=SampleKind.STATIC,
            label=Label.A,
            config=config,
            raiz=raiz,
            signer_id="s01",
            session_id="2026-09-08-manana",
            condiciones=CONDICIONES,
        )
        assert isinstance(resultado, Guardada), resultado
        guardadas.append(resultado)
    return guardadas


# --------------------------------------------------------------------------- #
# El criterio de aceptación
# --------------------------------------------------------------------------- #


def test_se_graban_veinte_muestras_de_una_letra(tmp_path: Path) -> None:
    config = Config()

    guardadas = grabar_veinte(tmp_path, config)

    assert len(guardadas) == 20
    carpeta = tmp_path / "s01" / "2026-09-08-manana" / "A"
    assert sorted(ruta.name for ruta in carpeta.glob("*.json")) == [
        f"{indice:03d}.json" for indice in range(1, 21)
    ]


def test_las_features_se_re_derivan_identicas_a_las_del_pipeline(
    tmp_path: Path,
) -> None:
    """El corazón de la fase.

    Se extraen las features de lo que estaba en memoria al grabar y de lo que
    quedó en el archivo, y se exige que sean **el mismo objeto de valores**: los
    42 componentes de cada frame, el vector promedio, σ, las escalas, el canal de
    trayectoria y las 24×44 filas del canal dinámico.

    Si esto fallara, guardar landmarks crudos no serviría para nada: cambiar la
    normalización obligaría a regrabar con personas frente a la cámara, que es
    exactamente lo que `ARQUITECTURA.md` §4.7 quiere evitar.
    """
    config = Config()
    guardadas = grabar_veinte(tmp_path, config)

    for indice, guardada in enumerate(guardadas):
        en_memoria = extract_sequence_features(
            still_sequence(
                translated(canonical_hand(), dx=3.0 * indice, dy=-2.0 * indice),
                length=config.capture.static_frames,
            ),
            config,
        )
        desde_disco = extract_sequence_features(
            read_sample(guardada.path).to_sample().sequence, config
        )

        assert desde_disco == en_memoria, guardada.path


def test_la_sigma_anotada_es_la_que_se_re_deriva(tmp_path: Path) -> None:
    """σ es un agregado de las 42 componentes sobre todos los frames: si un solo
    landmark hubiera perdido un bit al serializarse, este número cambia. Es la
    comprobación que hace `lsm-capture verificar` sobre el dataset real."""
    config = Config()
    grabar_veinte(tmp_path, config)

    for almacenada in iter_samples(tmp_path):
        extraccion = extract_sequence_features(almacenada.to_sample().sequence, config)
        assert not isinstance(extraccion, ExtractionRejected)
        assert extraccion.static.dispersion == almacenada.metadata.dispersion


def test_el_subcomando_verificar_aprueba_un_dataset_recien_grabado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    grabar_veinte(tmp_path, Config())

    codigo = main(["--raiz", str(tmp_path), "verificar"])

    assert codigo == 0
    assert "20 muestras releídas" in capsys.readouterr().out


def test_verificar_denuncia_una_muestra_manipulada(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Si alguien edita un landmark a mano —o si un día el formato perdiera
    precisión al escribirse— la σ re-derivada deja de coincidir con la que se
    anotó al aceptar la muestra, y el comando lo dice con nombre y apellidos."""
    guardadas = grabar_veinte(tmp_path, Config())
    ruta = guardadas[0].path
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["frames"][0]["landmarks"][8][0] += 0.05  # se movió la punta del índice
    ruta.write_text(json.dumps(payload), encoding="utf-8")

    codigo = main(["--raiz", str(tmp_path), "verificar"])

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "sigma re-derivada" in salida
    assert ruta.name in salida


def grabar_una_con_jitter(raiz: Path, config: Config) -> Guardada:
    """Una muestra con σ > 0, que es lo que `still_sequence` no da.

    Veinte frames idénticos tienen dispersión exactamente cero, y sobre cero no
    se puede hablar de un desplazamiento de unos pocos ULPs. El jitter por frame
    de `synthetic_samples` existe justo para esto.
    """
    (muestra,) = synthetic_samples(
        ("A",),
        signers=1,
        sessions=1,
        repetitions=1,
        frames=config.capture.static_frames,
    )
    resultado = guardar_muestra(
        frames=tuple(
            BufferedFrame(slot=frame, luminance=0.40)
            for frame in muestra.sequence.frames
        ),
        kind=SampleKind.STATIC,
        label=Label.A,
        config=config,
        raiz=raiz,
        signer_id="s01",
        session_id="2026-09-08-manana",
        condiciones=CONDICIONES,
    )
    assert isinstance(resultado, Guardada), resultado
    return resultado


def anotar_sigma(ruta: Path, valor: float) -> None:
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    payload["dispersion"] = valor
    ruta.write_text(json.dumps(payload), encoding="utf-8")


def test_verificar_tolera_el_ruido_de_libm_entre_plataformas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """σ se anota en el host, donde está la cámara, y se verifica en WSL.

    `atan2`, `sin` y `cos` no están correctamente redondeadas por IEEE 754, así
    que la libm de MSVC y la de glibc discrepan en el último bit y σ hereda la
    discrepancia. Medido sobre las 613 muestras de s01: exactas en Windows, entre
    1 y 10 ULPs de diferencia en Linux. Denunciarlo sería denunciar la
    arquitectura que `docker/README.md` §b eligió a propósito.
    """
    config = Config()
    guardada = grabar_una_con_jitter(tmp_path, config)
    anotada = json.loads(guardada.path.read_text(encoding="utf-8"))["dispersion"]
    assert anotada > 0.0, "sin jitter no hay ULPs de los que hablar"

    desplazada = anotada
    for _ in range(10):
        desplazada = math.nextafter(desplazada, math.inf)
    anotar_sigma(guardada.path, desplazada)

    codigo = main(["--raiz", str(tmp_path), "verificar"])

    assert codigo == 0
    assert "todas re-derivan" in capsys.readouterr().out


def test_verificar_denuncia_una_perdida_real_de_precision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """La tolerancia no puede tapar el fallo para el que existe el comando.

    2.3e-5 relativo es lo que se mueve σ si los landmarks se truncan a seis
    decimales —el «achiquemos los archivos» que este verificador tiene que
    cazar—, siete órdenes de magnitud por encima de la tolerancia.
    """
    config = Config()
    guardada = grabar_una_con_jitter(tmp_path, config)
    anotada = json.loads(guardada.path.read_text(encoding="utf-8"))["dispersion"]
    anotar_sigma(guardada.path, anotada * (1.0 + 2.3e-5))

    codigo = main(["--raiz", str(tmp_path), "verificar"])

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "sigma re-derivada" in salida
    assert guardada.path.name in salida


def test_verificar_un_dataset_vacio_no_es_un_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = main(["--raiz", str(tmp_path), "verificar"])

    assert codigo == 0
    assert "no hay muestras" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Lo que no se guarda
# --------------------------------------------------------------------------- #


def test_una_ventana_inestable_no_llega_al_dataset(tmp_path: Path) -> None:
    """`guardar_muestra` devuelve el motivo en vez de escribir. El rechazo es un
    valor de retorno y no una excepción porque en una sesión de captura rechazar
    es lo normal, no lo excepcional."""
    config = Config()
    frames = tuple(
        BufferedFrame(slot=frame, luminance=0.4)
        for frame in moving_sequence(
            canonical_hand(),
            arc_offsets(config.capture.static_frames),
        ).frames[:2]
    )

    resultado = guardar_muestra(
        frames=frames,
        kind=SampleKind.STATIC,
        label=Label.A,
        config=config,
        raiz=tmp_path,
        signer_id="s01",
        session_id="s",
        condiciones=CONDICIONES,
    )

    assert resultado is Rejection.TOO_FEW_FRAMES
    assert list(iter_samples(tmp_path)) == []


def test_una_ventana_con_huecos_no_llega_al_dataset(tmp_path: Path) -> None:
    config = Config()
    limpias = repeticion(config, 0)
    con_hueco = (
        *limpias[:5],
        BufferedFrame(slot=InvalidFrame(reason=InvalidReason.NO_HAND), luminance=0.4),
        *limpias[5:],
    )

    resultado = guardar_muestra(
        frames=con_hueco,
        kind=SampleKind.STATIC,
        label=Label.A,
        config=config,
        raiz=tmp_path,
        signer_id="s01",
        session_id="s",
        condiciones=CONDICIONES,
    )

    assert resultado is Rejection.HAS_GAPS
    assert list(iter_samples(tmp_path)) == []


# --------------------------------------------------------------------------- #
# Secuencias dinámicas
# --------------------------------------------------------------------------- #


def test_una_secuencia_dinamica_se_graba_con_su_trazo_entero(tmp_path: Path) -> None:
    """El caso que separa una J de una I.

    Se comprueba que el canal de trayectoria del §3.1 sobrevive al viaje a disco:
    si se hubieran guardado features en vez de landmarks crudos, o si la
    traslación del paso 3 se hubiera aplicado antes de guardar, `τ` saldría
    idénticamente cero y la J sería indistinguible de la I.
    """
    config = Config()
    trazo = moving_sequence(canonical_hand(), arc_offsets(30))
    frames = tuple(BufferedFrame(slot=frame, luminance=0.4) for frame in trazo.frames)

    resultado = guardar_muestra(
        frames=frames,
        kind=SampleKind.DYNAMIC,
        label=Label.J,
        config=config,
        raiz=tmp_path,
        signer_id="s01",
        session_id="s",
        condiciones=CONDICIONES,
    )

    assert isinstance(resultado, Guardada)
    almacenada = read_sample(resultado.path)
    assert almacenada.metadata.kind is SampleKind.DYNAMIC
    assert len(almacenada.frames) == 30

    extraccion = extract_sequence_features(almacenada.to_sample().sequence, config)
    assert not isinstance(extraccion, ExtractionRejected)
    assert extraccion == extract_sequence_features(trazo, config)
    assert any(punto != (0.0, 0.0) for punto in extraccion.trajectory.points)


def test_una_dinamica_demasiado_corta_se_rechaza(tmp_path: Path) -> None:
    """`config.py` valida que `dynamic_min_frames` no baje de
    `dtw.min_source_frames`, así que lo que se acepta aquí siempre se puede
    remuestrear a las 24 filas del §3.2."""
    config = Config()
    trazo = moving_sequence(canonical_hand(), arc_offsets(4))
    frames = tuple(BufferedFrame(slot=frame, luminance=0.4) for frame in trazo.frames)

    resultado = guardar_muestra(
        frames=frames,
        kind=SampleKind.DYNAMIC,
        label=Label.J,
        config=config,
        raiz=tmp_path,
        signer_id="s01",
        session_id="s",
        condiciones=CONDICIONES,
    )

    assert resultado is Rejection.TOO_FEW_FRAMES


# --------------------------------------------------------------------------- #
# Modo por defecto y selección de letras
# --------------------------------------------------------------------------- #


def test_el_glosario_decide_el_modo_inicial_de_cada_letra() -> None:
    assert _modo_por_defecto(Label.A) is SampleKind.STATIC
    assert _modo_por_defecto(Label.J) is SampleKind.DYNAMIC
    assert _modo_por_defecto(Label.ENIE) is SampleKind.DYNAMIC
    assert _modo_por_defecto(Label.DOBLE_R) is SampleKind.DYNAMIC


def test_la_clase_negativa_arranca_en_estatica() -> None:
    """`NONE` no está en el glosario y se graba de las dos formas —mano relajada,
    mano en tránsito—, así que preguntarle al glosario sería un `KeyError`."""
    assert _modo_por_defecto(Label.NONE) is SampleKind.STATIC


def test_por_defecto_se_graba_el_alfabeto_entero_mas_la_clase_negativa() -> None:
    """`NONE` va incluida a propósito y no como extra opcional: sin ella el
    clasificador asigna una de las 29 letras aunque la persona se esté rascando la
    nariz, y es la clase que todo el mundo olvida grabar."""
    letras = _resolver_letras(None)

    assert letras == (*ALPHABET, Label.NONE)
    assert Label.NONE in letras


def test_se_puede_grabar_un_subconjunto_de_letras() -> None:
    assert _resolver_letras("a, ENIE ,doble_r") == (
        Label.A,
        Label.ENIE,
        Label.DOBLE_R,
    )


def test_una_letra_inexistente_se_rechaza_con_la_lista_de_validas() -> None:
    with pytest.raises(SystemExit, match="letra desconocida"):
        _resolver_letras("A,Ñ")


# --------------------------------------------------------------------------- #
# Consentimiento
# --------------------------------------------------------------------------- #


def test_el_subcomando_de_consentimiento_registra_el_permiso(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = main(
        [
            "--raiz",
            str(tmp_path),
            "consentimiento",
            "--firmante",
            "s01",
            "--video",
            "--referencia",
            "folio 3",
        ]
    )

    assert codigo == 0
    assert "folio 3" in capsys.readouterr().out


def test_sin_la_bandera_el_consentimiento_registrado_es_de_solo_landmarks(
    tmp_path: Path,
) -> None:
    """Registrarse no es autorizar video: son dos preguntas distintas y la
    segunda tiene su propia bandera."""
    main(["--raiz", str(tmp_path), "consentimiento", "--firmante", "s01"])

    assert may_store_video(tmp_path, "s01") is False


def test_pedir_video_sin_consentimiento_registrado_aborta(tmp_path: Path) -> None:
    """La primera de las dos llaves. Sin registro, `--guardar-video` no arranca la
    sesión: se para antes de abrir la cámara, no a mitad de grabar."""
    with pytest.raises(SystemExit, match="consentimiento"):
        _resolver_video(tmp_path, "s01", pedido=True)


def test_sin_pedir_video_no_se_guarda_video_aunque_haya_consentimiento(
    tmp_path: Path,
) -> None:
    """La segunda llave. Aun con permiso registrado, el video solo se guarda si se
    pide explícitamente en la línea de comandos: apagado por defecto significa
    apagado por defecto."""
    save_consent(
        tmp_path,
        Consent(signer_id="s01", video=True, fecha=datetime(2026, 9, 8, tzinfo=UTC)),
    )

    assert _resolver_video(tmp_path, "s01", pedido=False) is False
    assert _resolver_video(tmp_path, "s01", pedido=True) is True


def test_una_muestra_sin_video_lo_deja_explicitamente_nulo(tmp_path: Path) -> None:
    """`video: null` en el archivo, no ausencia del campo: una clave que falta se
    confunde con un archivo truncado."""
    resultado = grabar_veinte(tmp_path, Config())[0]

    assert read_sample(resultado.path).metadata.video is None


# --------------------------------------------------------------------------- #
# Los metadatos de convención (Bloque 1.1)
# --------------------------------------------------------------------------- #


def test_cada_muestra_registra_con_que_convencion_se_grabo(tmp_path: Path) -> None:
    """Sin esto, cambiar `mediapipe_reports_mirrored_handedness` a mitad del
    proyecto dejaría el dataset mezclado sin ningún síntoma: media canonizada
    hacia una mano, media hacia la contraria, todas con la misma pinta.

    Con el campo, la reparación es un filtro y un espejo en vez de una nueva ronda
    de grabaciones con todo el mundo.
    """
    config = Config()
    guardada = grabar_veinte(tmp_path, config)[0]

    meta = read_sample(guardada.path).metadata

    assert meta.handedness_swapped is (
        config.hands.mediapipe_reports_mirrored_handedness
    )
    assert meta.handedness_convention is HANDEDNESS_CONVENTION


def test_la_muestra_registra_el_arco_ademas_de_la_sigma(tmp_path: Path) -> None:
    """Los dos números viajan siempre, decida cuál decida en ese modo: al leer la
    primera matriz de confusión se podrá preguntar si las dinámicas peores eran
    las de trazo más corto."""
    guardada = grabar_veinte(tmp_path, Config())[0]

    meta = read_sample(guardada.path).metadata

    assert meta.arc_length is not None
    assert meta.arc_length >= 0.0
    assert meta.dispersion >= 0.0


# --------------------------------------------------------------------------- #
# Criterio dinámico en el camino de guardado (Bloque 2.7)
# --------------------------------------------------------------------------- #


def test_una_dinamica_sin_recorrido_no_llega_al_dataset(tmp_path: Path) -> None:
    """Igual que una estática inestable: se devuelve el motivo y no se escribe
    nada. Una "J" sin trazo en el dataset es una "I" con otra etiqueta."""
    config = Config()
    quieta = still_sequence(canonical_hand(), length=config.capture.dynamic_min_frames)
    frames = tuple(BufferedFrame(slot=frame, luminance=0.4) for frame in quieta.frames)

    resultado = guardar_muestra(
        frames=frames,
        kind=SampleKind.DYNAMIC,
        label=Label.J,
        config=config,
        raiz=tmp_path,
        signer_id="s01",
        session_id="s",
        condiciones=CONDICIONES,
    )

    assert resultado is Rejection.TRAJECTORY_TOO_SHORT
    assert list(iter_samples(tmp_path)) == []


# --------------------------------------------------------------------------- #
# Bloqueo de sesión formal (Bloque 3.8)
# --------------------------------------------------------------------------- #


def _glosario(tmp_path: Path, *, validado: bool) -> Path:
    """Copia del glosario real con la sección 5 firmada o en blanco."""
    original = (
        Path(__file__).resolve().parents[1] / "docs" / "glosario-lsm.md"
    ).read_text(encoding="utf-8")
    corte = original.index("## 5")
    filas = (
        "|2026-09-12|A. Pérez|intérprete LSM|todas|sin objeciones|\n"
        if validado
        else "||||||\n"
    )
    destino = tmp_path / "glosario.md"
    destino.write_text(
        original[:corte]
        + "## 5. Registro de validación\n\n"
        + "|Fecha|Revisor|Rol|Letras revisadas|Observaciones|\n|-|-|-|-|-|\n"
        + filas,
        encoding="utf-8",
    )
    return destino


def _argumentos_grabar(tmp_path: Path, glosario: Path, *prueba: str) -> list[str]:
    return [
        "--raiz",
        str(tmp_path),
        "--glosario",
        str(glosario),
        "grabar",
        "--firmante",
        "s01",
        "--sesion",
        "sesion-a",
        "--luz-nivel",
        "INDOOR",
        "--luz-direccion",
        "FRONTAL",
        "--distancia",
        "MEDIUM",
        *prueba,
    ]


def test_el_modo_formal_se_niega_con_la_seccion_5_vacia(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El bloqueo más caro de saltarse.

    Tres personas x 29 letras x dos sesiones contra un glosario que no ha revisado
    un intérprete es irreversible salvo regrabando: si una seña está mal
    transcrita, las ~60 repeticiones de esa letra son ruido etiquetado.

    Se comprueba **antes** de abrir la cámara, así que este test no necesita uno.
    """
    glosario = _glosario(tmp_path, validado=False)

    codigo = main(_argumentos_grabar(tmp_path, glosario))

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "sección 5" in salida
    assert "--sesion-prueba" in salida


def test_el_modo_prueba_no_exige_el_glosario_validado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """La salida legítima: ensayar el encuadre antes de citar a nadie.

    Con la sección 5 vacía, el modo prueba pasa el bloqueo del glosario y se para
    en el siguiente cerrojo —la calibración—, que es lo que demuestra que el
    primero no lo detuvo.
    """
    glosario = _glosario(tmp_path, validado=False)

    codigo = main(_argumentos_grabar(tmp_path, glosario, "--sesion-prueba"))

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "sección 5" not in salida
    assert "ninguna cámara calibrada" in salida


def test_con_el_glosario_validado_el_modo_formal_pasa_al_siguiente_cerrojo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Firmada la sección 5, el bloqueo del glosario deja de aplicar y aparece el
    de la calibración. Los dos cerrojos son independientes."""
    glosario = _glosario(tmp_path, validado=True)

    codigo = main(_argumentos_grabar(tmp_path, glosario))

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "sección 5" not in salida
    assert "ninguna cámara calibrada" in salida


def test_las_muestras_de_prueba_no_entran_al_dataset(tmp_path: Path) -> None:
    """`pruebas/` vive dentro de la raíz para compartir consentimiento y
    calibración, pero un nivel más abajo, de modo que el recorrido del dataset
    formal no lo alcanza. Sin esto, el modelo se entrenaría con las grabaciones
    que se hicieron para encuadrar la cámara.
    """
    config = Config()
    pruebas = trial_root(tmp_path)
    grabar_veinte(pruebas, config)

    assert len(list(iter_samples(pruebas))) == 20
    assert list(iter_samples(tmp_path)) == []


# --------------------------------------------------------------------------- #
# Bloqueo por calibración (Bloque 1.4)
# --------------------------------------------------------------------------- #


def test_sin_calibracion_grabar_no_llega_a_abrir_la_camara(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Un rechazo, no un aviso, y **antes** de encender el hardware.

    La lateralidad invertida no deja ningún rastro en el dataset: el modelo
    entrena bien y la precisión es idéntica. Un aviso en la terminal, antes de
    cuarenta minutos de grabación con otra persona esperando, no lo lee nadie.

    Que se rechace sin tocar la cámara es lo que hace que este test pueda existir
    en una máquina sin webcam ni OpenCV: si el comando llegara a abrir el
    dispositivo, aquí fallaría por `cv2` y no por el cerrojo. La comprobación fina
    —por cámara, con la resolución que el driver entregue— vive en
    `tests/test_io_calibration.py`.
    """
    glosario = _glosario(tmp_path, validado=True)

    codigo = main(_argumentos_grabar(tmp_path, glosario))

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "ninguna cámara calibrada" in salida
    assert "lsm-capture calibrar" in salida
    # Si hubiera abierto la cámara, el error sería el de las dependencias.
    assert "cv2" not in salida


def test_una_calibracion_registrada_queda_legible(tmp_path: Path) -> None:
    """Lo que `lsm-capture calibrar` deja escrito, y que `grabar` exige después."""
    camara = camera_key(index=0, width=1280, height=720, backend="V4L2")
    save_calibration(
        tmp_path,
        Calibration(
            camera=camara,
            swap_handedness=True,
            convention=HANDEDNESS_CONVENTION,
            fecha=datetime(2026, 9, 8, tzinfo=UTC),
            width=1280,
            height=720,
            confirmado_por="quien grabó",
        ),
    )

    from lsm.io.calibration import current_calibration

    vigente = current_calibration(tmp_path, camara, swap_handedness=True)

    assert vigente is not None
    assert vigente.width == 1280
    assert vigente.confirmado_por == "quien grabó"
