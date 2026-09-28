"""La mano declarada y la etiqueta de MediaPipe: cuándo avisar, y solo avisar.

Desde `FEATURE_SPEC_VERSION` 2 la mano que canoniza el paso 2 de
`feature-spec.md` es la que **declara** quien firma al abrir la sesión. La
etiqueta de MediaPipe se sigue guardando, pero no decide: con la palma de lado
cambia de opinión a mitad de un trazo (ADR 0017), y una votación automática
sobre ella volcaría el trazo entero en silencio cuando fallara.

Queda un caso en que la etiqueta sí dice algo útil: que quien firma **cambió de
mano** sin volver a declararla. Eso se ve como una contradicción sostenida, con
score alto y con la mano **quieta** —donde MediaPipe acierta—, no como los
parpadeos de un trazo. Este módulo detecta ese caso para que el preview pregunte
«¿cambiaste de mano?». **Nunca cambia la mano por su cuenta**: la declaración es
de la persona.

Código puro, como `segmentation.py`: sin cámara ni disco. La tasa entra como
argumento para convertir `hands.mismatch_ms` a cuadros.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from lsm.config import Config
from lsm.segmentation import frames_from_ms, window_velocity
from lsm.types import FrameSlot, Handedness, RawFrame


def input_looks_unmirrored(frame: RawFrame, raised: Handedness) -> bool:
    """Si la mano levantada aparece del lado que le toca en la imagen sin espejar.

    Frente a una cámara, el lado derecho de una persona queda a la **izquierda**
    de la imagen tal como sale del sensor, y el izquierdo a la derecha. Con la
    mano levantada junto a su hombro, la muñeca tiene que caer en esa mitad. Si
    cae en la otra, algo espejó el cuadro antes del detector (§0.3).

    No usa la etiqueta de MediaPipe: esto es lo que la calibración comprueba en
    su lugar desde el ADR 0017.
    """
    muneca_x = frame.landmarks[0].x
    if raised is Handedness.RIGHT:
        return muneca_x < 0.5
    return muneca_x > 0.5


@dataclass
class HandMismatchWatcher:
    """Vigila si MediaPipe contradice a la mano declarada de forma sostenida.

    Se alimenta un cuadro a la vez. `observe` devuelve si hay que mostrar el
    aviso en ese cuadro.
    """

    config: Config
    declared: Handedness
    fps: float
    _run: int = field(default=0, init=False)
    _recent: deque[RawFrame] = field(default_factory=deque, init=False, repr=False)

    @property
    def frames_needed(self) -> int:
        return frames_from_ms(self.config.hands.mismatch_ms, self.fps)

    def observe(self, slot: FrameSlot) -> bool:
        if not isinstance(slot, RawFrame):
            self._run = 0
            self._recent.clear()
            return False

        # La misma velocidad que la máquina de estados (§6.1, v5).
        ventana = frames_from_ms(self.config.segmentation.velocity_window_ms, self.fps)
        self._recent.append(slot)
        while len(self._recent) > ventana + 1:
            self._recent.popleft()
        velocidad = window_velocity(self._recent, ventana, self.config)
        quieta = (
            velocidad is not None
            and velocidad < self.config.segmentation.velocity_threshold_per_s / self.fps
        )
        contradice = (
            slot.detected_handedness is not None
            and slot.detected_handedness is not self.declared
            and slot.handedness_score >= self.config.hands.mismatch_min_score
        )
        self._run = self._run + 1 if quieta and contradice else 0
        return self._run >= self.frames_needed
