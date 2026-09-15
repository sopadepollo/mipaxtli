"""La frontera con MediaPipe, sin MediaPipe.

Toda la suite tiene que correr en una máquina sin webcam y sin MediaPipe
instalado. Este archivo verifica dos cosas: que el detector falso reproduzca
secuencias grabadas fielmente —huecos incluidos—, y que el paquete siga sin
importar MediaPipe en ningún módulo.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from lsm.io.hands import (
    FIXTURE_SCHEMA_VERSION,
    FakeHandDetector,
    HandDetector,
    MediaPipeHandDetector,
    dump_frame_stream,
    load_frame_stream,
)
from lsm.synthetic import arc_offsets, canonical_hand, moving_sequence
from lsm.types import FrameStream, Handedness, InvalidFrame, InvalidReason, RawFrame

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sequences"


def a_stream() -> FrameStream:
    sequence = moving_sequence(canonical_hand(), arc_offsets(5))
    return (
        sequence.frames[0],
        sequence.frames[1],
        InvalidFrame(reason=InvalidReason.NO_HAND, detail="la mano salió del encuadre"),
        *sequence.frames[2:],
    )


def test_el_detector_falso_satisface_el_protocolo() -> None:
    assert isinstance(FakeHandDetector(slots=a_stream()), HandDetector)


def test_reproduce_los_frames_en_orden() -> None:
    stream = a_stream()
    detector = FakeHandDetector(slots=stream)

    assert tuple(detector.stream()) == stream


def test_detect_ignora_la_imagen_y_avanza_la_grabacion() -> None:
    """El doble no mira la imagen: reproduce lo grabado. Es lo que permite
    ejercitar la tubería completa sin cámara."""
    detector = FakeHandDetector(slots=a_stream())

    first = detector.detect(image=None)
    second = detector.detect(image=None)

    assert first is not second
    assert isinstance(first, RawFrame)


def test_al_agotarse_la_grabacion_devuelve_frames_invalidos() -> None:
    """Se acabó el video: no hay mano. Marcador explícito, no una excepción ni
    un `None`, para que el consumidor no necesite un caso especial."""
    detector = FakeHandDetector(slots=(a_stream()[0],))

    detector.detect(image=None)
    exhausted = detector.detect(image=None)

    assert isinstance(exhausted, InvalidFrame)
    assert exhausted.reason is InvalidReason.NO_HAND


def test_el_hueco_se_conserva_al_ida_y_vuelta_por_disco(tmp_path: Path) -> None:
    """Si el hueco se perdiera al guardar, una secuencia interrumpida se
    convertiría en una continua y el dataset mentiría."""
    stream = a_stream()
    path = tmp_path / "seq.json"

    dump_frame_stream(stream, path)
    restored = load_frame_stream(path)

    assert len(restored) == len(stream)
    assert isinstance(restored[2], InvalidFrame)
    assert restored[2].reason is InvalidReason.NO_HAND
    assert restored == stream


def test_los_landmarks_sobreviven_el_ida_y_vuelta_sin_perder_precision(
    tmp_path: Path,
) -> None:
    stream = a_stream()
    path = tmp_path / "seq.json"

    dump_frame_stream(stream, path)
    restored = load_frame_stream(path)

    original = stream[0]
    recovered = restored[0]
    assert isinstance(original, RawFrame)
    assert isinstance(recovered, RawFrame)
    for before, after in zip(original.landmarks, recovered.landmarks, strict=True):
        assert after.x == before.x
        assert after.y == before.y
        assert after.z == before.z


def test_un_fixture_de_otra_version_se_rechaza(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    dump_frame_stream(a_stream(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = FIXTURE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_frame_stream(path)


def test_el_fixture_de_ejemplo_del_repositorio_se_carga() -> None:
    stream = load_frame_stream(FIXTURES / "ejemplo_trazo_j.json")

    assert len(stream) >= 8
    assert any(isinstance(slot, InvalidFrame) for slot in stream)
    assert all(
        slot.handedness is Handedness.RIGHT
        for slot in stream
        if isinstance(slot, RawFrame)
    )


def test_desde_un_fixture_se_construye_el_detector() -> None:
    detector = FakeHandDetector.from_fixture(FIXTURES / "ejemplo_trazo_j.json")

    assert len(tuple(detector.stream())) >= 8


def test_ningun_modulo_del_paquete_importa_mediapipe() -> None:
    """`CLAUDE.md` §3: MediaPipe solo puede vivir en `io/hands.py`.

    Y ni siquiera ahí al importarlo. `MediaPipeHandDetector` hace el `import`
    dentro de `open()`, y `io/camera.py` y `io/preview.py` hacen lo mismo con
    `cv2`, precisamente para que este test pueda existir: la suite, el
    entrenamiento y la evaluación tienen que correr en una máquina sin webcam y
    sin las dependencias nativas de MediaPipe instaladas.

    **Importar el CLI de captura tampoco debe arrastrarlas.** Es el módulo que más
    cerca está del hardware, así que es donde una importación descuidada rompería
    esto sin que nadie lo notara hasta que fallara CI.
    """
    for module in (
        "lsm.capture",
        "lsm.classifiers.base",
        "lsm.classifiers.dummy",
        "lsm.classifiers.static_knn",
        "lsm.cli.capture",
        "lsm.cli.demo",
        "lsm.cli.evaluate",
        "lsm.cli.train",
        "lsm.config",
        "lsm.evaluation",
        "lsm.features",
        "lsm.io.camera",
        "lsm.io.corpus",
        "lsm.io.dataset",
        "lsm.io.hands",
        "lsm.io.preview",
        "lsm.segmentation",
        "lsm.spelling",
        "lsm.synthetic",
        "lsm.types",
        "lsm.vocabulary",
    ):
        importlib.import_module(module)

    assert "mediapipe" not in sys.modules
    assert "cv2" not in sys.modules
    assert "numpy" not in sys.modules


def test_el_detector_real_satisface_el_protocolo() -> None:
    """Construirlo no carga el modelo ni importa MediaPipe: eso es `open()`.

    La separación importa porque el constructor se ejecuta antes de saber si hay
    cámara, y fallar ahí por un modelo ausente daría un error confuso en un sitio
    que no tiene nada que ver.
    """
    detector = MediaPipeHandDetector(
        model_path=Path("data/models/hand_landmarker.task"), min_detection_score=0.5
    )

    assert isinstance(detector, HandDetector)
    assert "mediapipe" not in sys.modules


def test_el_detector_real_avisa_si_falta_el_modelo(tmp_path: Path) -> None:
    """El bundle pesa ~8 MB y no se versiona: el error tiene que decir qué hacer,
    no solo que no encontró un archivo."""
    detector = MediaPipeHandDetector(
        model_path=tmp_path / "no-esta.task", min_detection_score=0.5
    )

    with pytest.raises(FileNotFoundError, match="make model"):
        detector.open()


def test_detectar_sin_abrir_es_un_error_de_programacion() -> None:
    detector = MediaPipeHandDetector(
        model_path=Path("data/models/hand_landmarker.task"), min_detection_score=0.5
    )

    with pytest.raises(RuntimeError, match="open"):
        detector.detect(image=None)


def test_un_paso_de_timestamp_nulo_se_rechaza() -> None:
    """El modo VIDEO de MediaPipe exige timestamps estrictamente crecientes: con
    paso cero, el segundo cuadro llegaría con la misma marca que el primero."""
    with pytest.raises(ValueError, match="frame_interval_ms"):
        MediaPipeHandDetector(
            model_path=Path("x.task"), min_detection_score=0.5, frame_interval_ms=0
        )


# --------------------------------------------------------------------------- #
# Traducción de la salida de MediaPipe
#
# Se ejercita `_to_slot` directamente, con dobles que tienen la forma exacta de
# `HandLandmarkerResult`. Es un método privado y aun así es lo que más merece un
# test de todo el adaptador: es donde se decide qué mano se usa y con qué
# lateralidad, y equivocarse ahí no rompe nada de forma visible — corrompe el
# dataset entero en silencio.
#
# Los dobles evitan depender de MediaPipe, que no está instalado en el entorno de
# la suite a propósito. La forma que reproducen se verificó contra MediaPipe 1.0.1:
# `handedness: list[list[Category]]` con `category_name` y `score`, y
# `hand_landmarks: list[list[NormalizedLandmark]]` con `x`, `y`, `z`.
# --------------------------------------------------------------------------- #


@dataclass
class _Categoria:
    """Un `mediapipe.tasks.python.components.containers.Category`, en lo que importa."""

    category_name: str
    score: float


@dataclass
class _Punto:
    """Un `NormalizedLandmark`."""

    x: float
    y: float
    z: float


@dataclass
class _Resultado:
    """Un `HandLandmarkerResult`."""

    handedness: list[list[_Categoria]]
    hand_landmarks: list[list[_Punto]]


def _puntos(offset: float = 0.0) -> list[_Punto]:
    return [
        _Punto(x=0.5 + 0.01 * indice + offset, y=0.5 - 0.01 * indice, z=0.001 * indice)
        for indice in range(21)
    ]


def _detector(**cambios: object) -> MediaPipeHandDetector:
    opciones: dict[str, object] = {
        "model_path": Path("data/models/hand_landmarker.task"),
        "min_detection_score": 0.5,
    }
    opciones.update(cambios)
    return MediaPipeHandDetector(**opciones)  # type: ignore[arg-type]


def test_con_varias_manos_se_queda_con_la_de_mayor_score() -> None:
    """`feature-spec.md` §0.3: el alfabeto dactilológico de LSM es monomanual.

    Es también la razón de que `hands.num_hands` valga 2 y no 1: para poder
    elegir la mejor hay que ver más de una.
    """
    resultado = _Resultado(
        handedness=[[_Categoria("Left", 0.61)], [_Categoria("Left", 0.97)]],
        hand_landmarks=[_puntos(), _puntos(0.2)],
    )

    slot = _detector()._to_slot(resultado, width=1280, height=720)

    assert isinstance(slot, RawFrame)
    assert slot.detection_score == 0.97
    assert slot.landmarks[0].x == pytest.approx(0.7)  # la segunda mano


def test_el_interruptor_de_lateralidad_hace_exactamente_lo_que_dice() -> None:
    """El mecanismo, probado en las dos direcciones y sin depender del valor
    por defecto.

    Cuál de las dos sea la correcta **no se decide aquí**: se mide contra una
    cámara con `lsm-capture calibrar` y se registra. Con MediaPipe 1.0.1 resultó
    ser `swap_handedness=False` —devuelve la mano anatómica ante un cuadro sin
    espejar— pero eso es un hecho sobre una versión de una librería, no una
    propiedad del código, y este test seguiría valiendo si cambiara.

    Lo que sí es del código: el paso 2 de `feature-spec.md` espeja en X según este
    valor, así que equivocarse canoniza **todas** las muestras hacia la mano
    contraria, el vector queda coherente consigo mismo y nada falla hasta que
    alguien firma con la otra mano.
    """
    resultado = _Resultado(
        handedness=[[_Categoria("Left", 0.9)]], hand_landmarks=[_puntos()]
    )

    tal_cual = _detector(swap_handedness=False)._to_slot(
        resultado, width=640, height=480
    )
    invertido = _detector(swap_handedness=True)._to_slot(
        resultado, width=640, height=480
    )

    assert isinstance(tal_cual, RawFrame)
    assert isinstance(invertido, RawFrame)
    assert tal_cual.handedness is Handedness.LEFT
    assert invertido.handedness is Handedness.RIGHT


def test_por_defecto_no_se_invierte_la_lateralidad() -> None:
    """Medido el 2026-09-08 contra MediaPipe 1.0.1 y una webcam real.

    Va aparte del test anterior a propósito: aquel prueba el mecanismo, éste fija
    el valor que se eligió. Si algún día una versión de MediaPipe cambia la
    convención, es este el que hay que tocar — y el que obliga a acordarse de
    recalibrar.
    """
    detector = _detector()

    assert detector.swap_handedness is False


def test_un_score_bajo_se_distingue_de_la_ausencia_de_mano() -> None:
    """El motivo importa: cambia qué hace la máquina de estados y qué se escribe
    en el archivo. "No había nadie" y "había alguien y no me fío" son cosas
    distintas."""
    resultado = _Resultado(
        handedness=[[_Categoria("Right", 0.3)]], hand_landmarks=[_puntos()]
    )

    slot = _detector()._to_slot(resultado, width=640, height=480)

    assert isinstance(slot, InvalidFrame)
    assert slot.reason is InvalidReason.LOW_DETECTION_SCORE


def test_un_resultado_vacio_es_un_frame_sin_mano() -> None:
    slot = _detector()._to_slot(
        _Resultado(handedness=[], hand_landmarks=[]), width=640, height=480
    )

    assert isinstance(slot, InvalidFrame)
    assert slot.reason is InvalidReason.NO_HAND


def test_una_lateralidad_desconocida_no_se_adivina() -> None:
    """Si MediaPipe devolviera una categoría que no es Left ni Right, inventar una
    sería peor que descartar el frame: el paso 2 espeja según ese valor."""
    resultado = _Resultado(
        handedness=[[_Categoria("Unknown", 0.9)]], hand_landmarks=[_puntos()]
    )

    slot = _detector()._to_slot(resultado, width=640, height=480)

    assert isinstance(slot, InvalidFrame)
    assert "Unknown" in slot.detail


def test_un_numero_raro_de_landmarks_se_descarta() -> None:
    """Los 21 índices son estructura, no una sugerencia: `RawFrame` los exige y el
    paso 7 depende del orden. Mejor descartar el frame que reventar al construirlo.
    """
    resultado = _Resultado(
        handedness=[[_Categoria("Right", 0.9)]], hand_landmarks=[_puntos()[:15]]
    )

    slot = _detector()._to_slot(resultado, width=640, height=480)

    assert isinstance(slot, InvalidFrame)
    assert "15" in slot.detail


def test_las_dimensiones_del_cuadro_viajan_con_el_frame() -> None:
    """El paso 1 divide por la relación de aspecto. Asumir 1280x720 cuando la
    cámara entregó 640x480 deforma la mano en silencio."""
    resultado = _Resultado(
        handedness=[[_Categoria("Right", 0.9)]], hand_landmarks=[_puntos()]
    )

    slot = _detector()._to_slot(resultado, width=640, height=480)

    assert isinstance(slot, RawFrame)
    assert (slot.width, slot.height) == (640, 480)
    assert slot.aspect_ratio == pytest.approx(640 / 480)
