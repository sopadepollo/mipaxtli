"""El criterio de aceptación de la Fase 3, sin cámara.

> Deletrear una palabra de cinco letras **sin errores de segmentación**.

Lo que se exige aquí es **segmentación, no acierto**: que cinco señas produzcan
cinco símbolos. Que la máquina no parta una seña en dos, no funda dos en una, no
repita la que sigue sostenida y no escriba nada mientras la mano viaja.

La distinción no es un tecnicismo. Con el accuracy medido en la Fase 2 (0.9261)
una palabra de cinco letras sale entera el 68% de las veces, así que un test que
dependiera del acierto del clasificador fallaría una de cada tres ejecuciones sin
que nada estuviera roto — y un test intermitente se acaba ignorando, que es
exactamente como pasó desapercibida la deriva del glosario que cuenta el README.
Lo que estas secuencias sintéticas ejercitan sí es determinista: dos ejecuciones
producen los mismos bits.

## El clasificador de estas pruebas

Es un `StaticKnnClassifier` **entrenado de verdad** sobre el corpus sintético, no
un doble que devuelva etiquetas de una lista. El motivo es que la máquina de
estados llama a `classify` también en las ventanas que va a rechazar —por margen
bajo, por letra repetida— y una lista consumida por llamada se desincronizaría en
cada rechazo: el test pasaría o fallaría por razones que no son las que dice
medir. Un clasificador que mira la ventana devuelve lo mismo cada vez que se le
enseña la misma mano, que es la propiedad que hace falta.

`lsm.synthetic.class_hand(ordinal)` da una configuración de mano distinta y
estable por clase, y el orden de `ETIQUETAS` es el que asocia cada ordinal con su
letra.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lsm.classifiers.registry import ClassifierRegistry
from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.cli.demo import (
    Medida,
    Sesion,
    _build_parser,
    _medida_pedida,
    aplicar_evento,
    main,
)
from lsm.config import Config
from lsm.segmentation import (
    FrameThresholds,
    LetterEmitted,
    RejectionReason,
    SegmentationEvent,
    WindowDynamic,
    WindowRejected,
    frames_from_ms,
    run_segmentation,
)
from lsm.spelling import HandAbsent, HandPresent, SpellingState, render_text
from lsm.synthetic import class_hand, synthetic_samples, to_frame, translated
from lsm.types import FrameSlot, InvalidFrame, InvalidReason
from lsm.vocabulary import Label

RAIZ_REPO = Path(__file__).resolve().parents[1]

CONFIG = Config()

#: Los umbrales de `CONFIG` resueltos a cuadros con la tasa NOMINAL, que es la
#: que usa `run_segmentation` cuando nadie le pasa una medida — el caso de estos
#: tests, donde no hay cámara ni tasa real que medir.
UMBRALES = FrameThresholds.from_config(CONFIG, CONFIG.capture.camera_fps)

#: Las clases con las que se entrena. El **orden importa**: `synthetic_samples`
#: asigna la configuración de mano por posición, no por nombre, así que este es el
#: único sitio donde se decide qué mano es cada letra.
#:
#: `NONE` entra porque en vivo entra: la clase negativa es la que absorbe la mano
#: relajada y las transiciones, y entrenar sin ella dejaría a las cuatro letras
#: repartiéndose todo el espacio de features.
ETIQUETAS: tuple[Label, ...] = (Label.C, Label.A, Label.S, Label.N, Label.NONE)
_ORDINAL: dict[Label, int] = {label: i for i, label in enumerate(ETIQUETAS)}

PALABRA: tuple[Label, ...] = (Label.C, Label.A, Label.S, Label.A, Label.S)


def _entrenado() -> StaticKnnClassifier:
    classifier = StaticKnnClassifier(config=CONFIG)
    classifier.fit(
        list(
            synthetic_samples(
                tuple(label.value for label in ETIQUETAS),
                signers=3,
                sessions=2,
                repetitions=3,
            )
        )
    )
    return classifier


#: Se entrena una vez para todo el módulo: es determinista y no depende de nada
#: que un test pueda ensuciar.
CLASIFICADOR = _entrenado()
#: Lo que la demo pasa de verdad a la segmentación: el registry, con solo el
#: estático cargado. Los trazos que se colaran al camino dinámico saldrían UNKNOWN.
REGISTRO = ClassifierRegistry(static=CLASIFICADOR)

# --------------------------------------------------------------------------- #
# El ritmo de los frames
# --------------------------------------------------------------------------- #
#
# Los dos números de abajo son los que deciden si el criterio se puede comprobar,
# y ninguno es redondo por casualidad.

#: Frames de viaje entre dos letras.
#:
#: Tan largos como el buffer circular. La razón original —un viaje más corto
#: fundía dos manos en la ventana clasificada— desapareció con el ADR 0013, pero
#: el ritmo de `QUIETO` y la segunda oportunidad de las letras que acumulan
#: evidencia se calcularon con este viaje, y acortarlo mete el rebote dentro del
#: cooldown de la letra anterior.
#:
#: **Desde el ADR 0015 este viaje es un candidato dinámico**: 800 ms de
#: movimiento superan `motion_min`. El texto no cambia —el trazo se rechaza y la
#: letra de llegada sale por el camino estático—, pero cada viaje deja un
#: `LOW_CONFIDENCE` de la ruta dinámica y la letra de llegada sale con la quietud
#: que cerró el candidato. Los tests que miden latencia estática usan
#: `VIAJE_CORTO`.
VIAJE = UMBRALES.buffer_size

#: Un tránsito que se queda por debajo de `motion_min`: el viaje suma sus frames
#: más el salto de llegada a la quietud, y con `motion_min` frames móviles se
#: volvería candidato. Desde la v5 la velocidad de un frame es la de los últimos
#: `velocity_window_frames` (§6.1): tras una parada en seco la mano sigue
#: «moviéndose» `ARRASTRE` frames más, que también suman.
ARRASTRE = UMBRALES.velocity_window_frames - 1
VIAJE_CORTO = UMBRALES.motion_min_frames - 2 - ARRASTRE

#: Frames de mano quieta por letra, sumados término a término:
#:
#: - `motion_confirm_low_frames + 1 + ARRASTRE`: los primeros frames quietos
#:   gastan todavía la velocidad del viaje —uno, más los de la ventana—; los
#:   siguientes son la espera hasta la primera clasificación. Como `VIAJE` es un
#:   candidato dinámico, esa espera es el reposo que lo cierra
#:   —`motion_confirm_low_frames`— y no `stable_frames`: la letra sale en el
#:   frame siguiente al rechazo del trazo (ADR 0015).
#: - `+ emit_cooldown_frames`: el cooldown de EMIT, durante el cual no se clasifica.
#: - `+ stable_frames`: lo que tarda la ventana en volver a declararse estable al
#:   salir del cooldown.
#: - `+ 1` de margen, para que esa segunda ventana llegue a clasificarse **dentro**
#:   del bloque.
#:
#: Es decir: el bloque dura justo lo necesario para que la máquina tenga
#: **exactamente una** segunda oportunidad de escribir otra vez la letra que sigue
#: sostenida. Un bloque más corto haría pasar el test sin haber comprobado nunca
#: esa propiedad, que es la mitad del criterio. Uno más largo solo acumula más
#: rechazos por `REPEATED_LETTER`, ninguna emisión de más.
QUIETO = (
    (UMBRALES.motion_confirm_low_frames + 1 + ARRASTRE)
    + UMBRALES.emit_cooldown_frames
    + UMBRALES.stable_frames
    + 1
)


def quieto(label: Label, count: int = QUIETO) -> list[FrameSlot]:
    """La mano de `label` quieta en el mismo sitio: velocidad cero."""
    frame = to_frame(
        translated(class_hand(_ORDINAL[label]), 640.0, 400.0), width=1280, height=720
    )
    return [frame for _ in range(count)]


def viaje(label: Label, count: int = VIAJE, step: float = 25.0) -> list[FrameSlot]:
    """La mano viajando por el encuadre, muy por encima de `velocity_threshold`.

    Lleva **la configuración de la letra a la que va**, no la de la que deja. Es lo
    que ocurre al firmar —la mano forma la letra mientras se coloca, y en el rebote
    de una letra doble no cambia de forma en ningún momento— y es lo que hace que
    la ventana que se clasifica al detenerse no sea una mezcla de dos manos.
    """
    return [
        to_frame(
            translated(class_hand(_ORDINAL[label]), 300.0 + step * index, 400.0),
            width=1280,
            height=720,
        )
        for index in range(count)
    ]


def frames(
    labels: tuple[Label, ...], *, rebote: bool = True, largo_viaje: int = VIAJE
) -> list[FrameSlot]:
    """Un bloque quieto por letra, con el viaje entre ellas.

    Sin `rebote` las señas se pegan una a otra, que es lo que hace falta para
    comprobar la regla de letras dobles: dos N seguidas sin que la mano se mueva.
    """
    flujo: list[FrameSlot] = []
    for indice, label in enumerate(labels):
        if indice and rebote:
            flujo += viaje(label, count=largo_viaje)
        flujo += quieto(label)
    return flujo


def eventos(flujo: list[FrameSlot], config: Config = CONFIG) -> list[SegmentationEvent]:
    return list(run_segmentation(iter(flujo), config, REGISTRO))


def deletrear(flujo: list[FrameSlot], config: Config = CONFIG) -> SpellingState:
    """Corre la tubería entera y devuelve el estado del buffer de deletreo."""
    sesion = Sesion(config=config)
    for evento in run_segmentation(iter(flujo), config, REGISTRO):
        aplicar_evento(sesion, evento)
    return sesion.state


def con_presencia(flujo: list[FrameSlot], sesion: Sesion) -> Iterator[FrameSlot]:
    """Cede los frames aplicando presencia y ausencia, como hace la demo real.

    `run_segmentation` no emite un evento por frame, así que por esa vía la sesión
    no ve las ausencias: en `cli/demo.py` la presencia se aplica dentro del
    generador del flujo, un frame por vuelta. Esto lo replica, y testearlo aquí es
    lo que impide que ese mecanismo se rompa sin avisar.
    """
    for slot in flujo:
        sesion.aplicar(
            HandAbsent() if isinstance(slot, InvalidFrame) else HandPresent()
        )
        yield slot


def emitidas(eventos_: list[SegmentationEvent]) -> list[str]:
    return [e.prediction.label for e in eventos_ if isinstance(e, LetterEmitted)]


def rechazos(eventos_: list[SegmentationEvent]) -> list[RejectionReason]:
    return [e.reason for e in eventos_ if isinstance(e, WindowRejected)]


def rechazos_estaticos(eventos_: list[SegmentationEvent]) -> list[RejectionReason]:
    """Los rechazos del camino estático: sin los que cierran un trazo dinámico.

    Un rechazo de la ruta dinámica llega siempre justo detrás de su
    `WindowDynamic`, en el mismo frame; así se distinguen sin adivinar.
    """
    trazos = {e.frame_index for e in eventos_ if isinstance(e, WindowDynamic)}
    return [
        e.reason
        for e in eventos_
        if isinstance(e, WindowRejected) and e.frame_index not in trazos
    ]


# --------------------------------------------------------------------------- #
# El criterio
# --------------------------------------------------------------------------- #


def test_una_palabra_de_cinco_letras_produce_cinco_simbolos() -> None:
    """EL CRITERIO DE LA FASE.

    Cinco señas, cinco símbolos. Se comprueba la longitud antes que el texto a
    propósito: si el clasificador fallara una letra, lo que tiene que fallar es la
    aserción del texto y no la del criterio, para que quede claro cuál de las dos
    cosas se rompió.
    """
    estado = deletrear(frames(PALABRA))

    assert len(estado.word) == 5
    assert render_text(estado) == "casas"


def test_ninguna_sena_se_parte_en_dos_ni_se_funde_con_la_siguiente() -> None:
    """El criterio visto desde los eventos, que es donde se lee el porqué.

    Cinco emisiones, una por seña, y ni una de más. Las cinco ventanas que la
    máquina clasifica de nuevo sobre la mano todavía sostenida se rechazan por
    `REPEATED_LETTER`: es el cerrojo `pending_repeat` haciendo su trabajo, y es lo
    que impide que sostener una letra la escriba dos veces.
    """
    stream = eventos(frames(PALABRA))

    assert emitidas(stream) == ["C", "A", "S", "A", "S"]
    # Cada viaje de 800 ms es un candidato dinámico que se rechaza (no hay
    # modelo dinámico cargado): uno por viaje, y ninguno más. Ver `VIAJE`.
    dinamicos = [e for e in stream if isinstance(e, WindowDynamic)]
    assert len(dinamicos) == len(PALABRA) - 1
    assert set(rechazos_estaticos(stream)) == {RejectionReason.REPEATED_LETTER}
    # Una por seña. La `C` del corpus sintético se clasifica con 0.743, por
    # debajo de `high_confidence`, así que acumula evidencia hasta agotar la
    # ventana y emite en el frame 23 de su bloque en vez de en el 5. Con
    # `motion_confirm_low_ms` = 400 el bloque se le acababa antes de la segunda
    # oportunidad y eran cuatro; con 667 (v5, ADR 0017) `QUIETO` es más largo y
    # también la tiene. Ver la emisión progresiva en `lsm.segmentation` y
    # `docs/adr/0013-la-ventana-mezclada.md`.
    assert len(rechazos_estaticos(stream)) == len(PALABRA)


def test_la_mano_que_viaja_no_escribe_nada() -> None:
    """Mientras la velocidad supera el umbral la ventana nunca es estable, así que
    no se llega a clasificar. Sin esto, el texto se llenaría de letras basura entre
    seña y seña."""
    solo_viaje = [*viaje(Label.C, count=90)]

    assert render_text(deletrear(solo_viaje)) == ""


def test_la_letra_doble_exige_el_rebote() -> None:
    """`segmentation.py` no repite la misma letra sin que la mano supere
    `velocity_threshold` desde la emisión anterior.

    Sin rebote entre las dos N sale una sola; con rebote, dos. Es la regla que hace
    posible deletrear "carro" o "llave" sin convertir el deletreo en un ejercicio
    de sacudir la mano entre todas las letras.
    """
    doble = (Label.N, Label.N)

    assert render_text(deletrear(frames(doble, rebote=False))) == "n"
    assert render_text(deletrear(frames(doble, rebote=True))) == "nn"


def test_nada_por_debajo_del_umbral_llega_al_buffer() -> None:
    """`segmentation.min_confidence` es el piso que la máquina exige por encima del
    del propio clasificador. Puesto por las nubes, el texto queda vacío: ninguna
    ventana insuficientemente segura escribe."""
    exigente = Config.model_validate(
        # Los dos umbrales por las nubes: `config.py` no admite un
        # `high_confidence` por debajo del piso, y lo que se quiere aquí es que
        # NADA emita, ni de inmediato ni acumulando.
        {"segmentation": {"min_confidence": 0.99, "high_confidence": 0.99}}
    )
    stream = eventos(frames(PALABRA), exigente)

    assert emitidas(stream) == []
    assert set(rechazos(stream)) == {RejectionReason.LOW_CONFIDENCE}
    assert deletrear(frames(PALABRA), exigente).word == ()


def test_la_mano_abajo_entre_dos_palabras_pone_un_espacio_y_uno_solo() -> None:
    """El único gesto de control del proyecto: bajar la mano cierra la palabra.

    El hueco dura el doble del umbral para comprobar que el espacio se escribe una
    vez y no uno por frame — la mano abajo no es un evento, es un estado que dura.
    """
    sesion = Sesion(config=CONFIG)
    palabras = (Label.C, Label.A, Label.S, Label.A)
    hueco: list[FrameSlot] = [InvalidFrame(reason=InvalidReason.NO_HAND)] * (
        frames_from_ms(CONFIG.spelling.space_after_absent_ms, CONFIG.capture.camera_fps)
        * 2
    )
    flujo = [*frames(palabras[:2]), *hueco, *frames(palabras[2:])]

    for evento in run_segmentation(con_presencia(flujo, sesion), CONFIG, REGISTRO):
        aplicar_evento(sesion, evento)

    assert render_text(sesion.state) == "ca sa"


# --------------------------------------------------------------------------- #
# La medicion de fps: que pide cada combinacion de flags
# --------------------------------------------------------------------------- #


def _pedida(*argv: str) -> Medida | str | None:
    return _medida_pedida(_build_parser().parse_args(argv), CONFIG)


def test_sin_flags_no_se_mide_nada() -> None:
    """La instrumentacion del HUD corre siempre; la medicion de duracion fija,
    solo cuando se pide. Una sesion normal no debe terminarse sola al minuto."""
    assert _pedida() is None


def test_medir_fps_toma_la_duracion_de_la_configuracion() -> None:
    """`CLAUDE.md` §5: el default es un umbral y vive en `config.yaml`, no en el
    parser."""
    medida = _pedida("--medir-fps")

    assert isinstance(medida, Medida)
    assert medida.duracion == CONFIG.telemetry.benchmark_seconds
    assert medida.medicion.cuadros == 0


def test_medir_segundos_manda_sobre_la_configuracion() -> None:
    medida = _pedida("--medir-fps", "--medir-segundos", "12.5")

    assert isinstance(medida, Medida)
    assert medida.duracion == 12.5


def test_medir_fps_sobre_un_dataset_grabado_se_rechaza() -> None:
    """Lo que se mide es el bucle en vivo: cuanto tarda la camara en entregar un
    cuadro y cuanto tarda la tuberia en procesarlo. Sobre un dataset ya grabado
    no hay ninguna de las dos cosas, y devolver un numero de todas formas seria
    peor que negarse: se leeria como la tasa de la maquina."""
    error = _pedida("--medir-fps", "--desde-dataset", "data/raw/s01/x")

    assert isinstance(error, str)
    assert "--desde-dataset" in error


def test_medir_segundos_sin_medir_fps_se_rechaza() -> None:
    """Pedir una duracion sin pedir la medicion es un malentendido, y correr la
    demo normal en silencio lo dejaria sin resolver."""
    error = _pedida("--medir-segundos", "30")

    assert isinstance(error, str)
    assert "--medir-fps" in error


def test_una_duracion_negativa_se_rechaza() -> None:
    error = _pedida("--medir-fps", "--medir-segundos", "-1")

    assert isinstance(error, str)


def test_un_transito_corto_ya_no_funde_dos_manos_en_una_ventana() -> None:
    """El defecto medido del ADR 0013, fijado como regresion.

    Con un transito de 4 frames entre dos letras —mas corto que `buffer_size`—
    la ventana que la maquina declaraba estable sobre la letra nueva todavia
    llevaba dentro frames de la anterior: se clasificaba una mezcla de dos manos
    y salia una letra que nadie firmo.

    Este lo toca a proposito, con el transito mas corto que el ADR reporta haber
    medido.
    """
    corto = 4
    assert corto < UMBRALES.buffer_size

    flujo = [
        *quieto(Label.S),
        *viaje(Label.A, count=corto),
        *quieto(Label.A),
    ]

    assert render_text(deletrear(flujo)) == "sa"


def test_un_transito_largo_pasa_por_el_camino_dinamico_y_la_letra_sale_igual() -> None:
    """Un viaje de `buffer_size` frames (800 ms) supera `motion_min` y se vuelve
    candidato dinámico. Con la configuración de fábrica eso no cambia el texto:
    el trazo se rechaza —aquí no hay modelo dinámico; con él, el DTW lo rechaza
    por distancia— y la letra de llegada sale por el camino estático en el frame
    siguiente, con la quietud que cerró el candidato ya contada. Lo que sí cambia
    es la latencia: esa letra sale tras `motion_confirm_low_ms` de quietud y no
    tras `stable_ms`. Ver `docs/adr/0015-el-camino-dinamico-de-la-segmentacion.md`.
    """
    largo = UMBRALES.buffer_size
    assert largo + 1 >= UMBRALES.motion_min_frames
    flujo = [*quieto(Label.S), *viaje(Label.A, count=largo), *quieto(Label.A)]

    stream = eventos(flujo)

    assert emitidas(stream) == ["S", "A"]
    assert RejectionReason.LOW_CONFIDENCE in rechazos(stream)
    emision_a = [e for e in stream if isinstance(e, LetterEmitted)][1]
    assert len(emision_a.window) == UMBRALES.motion_confirm_low_frames + 1


def test_la_letra_segura_sale_rapido_y_la_dudosa_espera() -> None:
    """La latencia adaptativa del bloque 2, sobre el clasificador de verdad.

    `A` y `S` se resuelven por encima de `high_confidence` y salen con la
    ventana minima —`stable_frames` frames—; la `C`, que en este corpus se
    clasifica con 0.743, acumula evidencia hasta agotar la ventana. Rapido donde
    puede permitirselo, prudente solo donde hace falta.

    Y la `C` no es una letra cualquiera para este ejemplo: `C`/`O` es el par
    dominante de la matriz de confusion de la Fase 2
    (`docs/adr/0011-calibracion-de-la-fase-2.md`), es decir justo el caso en el
    que acumular evidencia vale lo que cuesta.
    """
    emisiones = [
        evento
        for evento in eventos(frames(PALABRA, largo_viaje=VIAJE_CORTO))
        if isinstance(evento, LetterEmitted)
    ]
    por_letra = {evento.prediction.label: len(evento.window) for evento in emisiones}

    assert por_letra["A"] == UMBRALES.stable_frames
    assert por_letra["S"] == UMBRALES.stable_frames
    assert por_letra["C"] == UMBRALES.buffer_size - 1


def test_una_ruta_sin_muestras_lo_dice_en_vez_de_callarse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Una linea en blanco y salida 0 es indistinguible de "no reconocio nada".

    Y el error es facil de cometer: la raiz que `iter_sample_paths` espera es la
    del dataset, y apuntar a una sesion concreta —que es lo que uno haria, y lo
    que documentaba `COMO-PROBAR`— no encuentra ninguna muestra.
    """
    codigo = main(
        [
            "--config",
            str(RAIZ_REPO / "config.yaml"),
            "--desde-dataset",
            str(tmp_path),
        ]
    )

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "no hay muestras" in salida
    assert "<firmante>/<sesion>/<letra>" in salida
