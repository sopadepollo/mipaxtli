"""Interfaces de linea de comandos."""

from __future__ import annotations

from typing import Final

from lsm.types import Handedness

#: Qué decir cuando faltan las dependencias opcionales. Ocurre siempre la primera
#: vez, porque `make setup` no las instala a propósito: la suite, el entrenamiento
#: y la evaluación corren sin cámara, y arrastrar MediaPipe a todos esos entornos
#: por dos comandos que solo se usan delante de una webcam sería un mal negocio.
#:
#: Vive aquí y no en `cli/capture.py` porque los comandos que abren cámara son
#: dos —`lsm-capture grabar` y `lsm-demo`— y el segundo no debe importar un
#: privado del primero para dar el mismo mensaje.
MENSAJE_SIN_EXTRAS: Final = (
    "faltan las dependencias de captura ({modulo}). Se instalan aparte porque el "
    "resto del proyecto no las necesita:\n"
    "  make setup-capture\n"
    "  make model"
)

#: Cómo se escribe la mano en `--mano`. En español porque es lo que teclea quien
#: firma, y cerrado a dos valores porque no hay otro: el alfabeto es monomanual.
MANOS: Final = {"derecha": Handedness.RIGHT, "izquierda": Handedness.LEFT}

#: Lo que dice la ayuda de `--mano` en los dos comandos que abren cámara.
AYUDA_MANO: Final = (
    "mano con la que firmas en esta sesión. Es la que canoniza el paso 2 del "
    "contrato (ADR 0017); la etiqueta de MediaPipe solo se guarda para "
    "diagnóstico. Para cambiar de mano, abre otra sesión."
)
