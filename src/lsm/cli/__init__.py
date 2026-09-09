"""Interfaces de linea de comandos."""

from __future__ import annotations

from typing import Final

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
