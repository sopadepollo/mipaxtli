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

from lsm.classifiers.static_knn import StaticKnnClassifier
from lsm.cli.demo import Sesion, aplicar_evento
from lsm.config import Config
from lsm.segmentation import (
    LetterEmitted,
    RejectionReason,
    SegmentationEvent,
    WindowRejected,
    run_segmentation,
)
from lsm.spelling import HandAbsent, HandPresent, SpellingState, render_text
from lsm.synthetic import class_hand, synthetic_samples, to_frame, translated
from lsm.types import FrameSlot, InvalidFrame, InvalidReason
from lsm.vocabulary import Label

CONFIG = Config()

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

# --------------------------------------------------------------------------- #
# El ritmo de los frames
# --------------------------------------------------------------------------- #
#
# Los dos números de abajo son los que deciden si el criterio se puede comprobar,
# y ninguno es redondo por casualidad.

#: Frames de viaje entre dos letras.
#:
#: Tan largos como el buffer circular **a propósito**. La ventana que se clasifica
#: son los últimos `buffer_size` frames, así que si el viaje fuera más corto, la
#: primera ventana que la máquina declare estable sobre la letra nueva todavía
#: llevaría dentro frames de la anterior: se clasificaría una mezcla de dos manos y
#: la máquina emitiría una letra que nadie hizo. Medido con un viaje de 4 frames,
#: eso es exactamente lo que pasa — la S salía como A.
VIAJE = CONFIG.segmentation.buffer_size

#: Frames de mano quieta por letra, sumados término a término:
#:
#: - `stable_frames + 1`: el primer frame quieto gasta todavía la velocidad del
#:   viaje; los `stable_frames` siguientes son los que la máquina exige para
#:   declarar la ventana estable. Ahí se emite la letra.
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
    (CONFIG.segmentation.stable_frames + 1)
    + CONFIG.segmentation.emit_cooldown_frames
    + CONFIG.segmentation.stable_frames
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


def frames(labels: tuple[Label, ...], *, rebote: bool = True) -> list[FrameSlot]:
    """Un bloque quieto por letra, con el viaje entre ellas.

    Sin `rebote` las señas se pegan una a otra, que es lo que hace falta para
    comprobar la regla de letras dobles: dos N seguidas sin que la mano se mueva.
    """
    flujo: list[FrameSlot] = []
    for indice, label in enumerate(labels):
        if indice and rebote:
            flujo += viaje(label)
        flujo += quieto(label)
    return flujo


def eventos(flujo: list[FrameSlot], config: Config = CONFIG) -> list[SegmentationEvent]:
    return list(run_segmentation(iter(flujo), config, CLASIFICADOR.predict))


def deletrear(flujo: list[FrameSlot], config: Config = CONFIG) -> SpellingState:
    """Corre la tubería entera y devuelve el estado del buffer de deletreo."""
    sesion = Sesion(config=config)
    for evento in run_segmentation(iter(flujo), config, CLASIFICADOR.predict):
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
    assert rechazos(stream) == [RejectionReason.REPEATED_LETTER] * 5


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
    exigente = Config.model_validate({"segmentation": {"min_confidence": 0.99}})
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
        CONFIG.spelling.space_after_absent_frames * 2
    )
    flujo = [*frames(palabras[:2]), *hueco, *frames(palabras[2:])]

    for evento in run_segmentation(
        con_presencia(flujo, sesion), CONFIG, CLASIFICADOR.predict
    ):
        aplicar_evento(sesion, evento)

    assert render_text(sesion.state) == "ca sa"
