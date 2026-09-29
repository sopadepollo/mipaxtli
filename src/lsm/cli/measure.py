"""`lsm-medir`: las mediciones de la serie de tolerancia a MediaPipe.

- `tolerancia` (Paso 1, ADR 0026): huecos por fase del intento, scores antes de
  cada pérdida, variación del largo de los huesos, ángulos articulares, saltos
  de la palma y temblor en reposo por dedo, sobre las sesiones de diagnóstico y
  las muestras del dataset. Es de donde salen los umbrales de los pasos 2 a 4.

No necesita cámara. Lee `data/diagnostico/` y `data/raw/`; el cálculo es
`lsm.measurements`, que es puro.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from lsm.config import Config, load_config
from lsm.io.dataset import iter_sample_paths, read_sample
from lsm.io.diagnostics import (
    DiagnosticSession,
    iter_diagnostic_folders,
    load_diagnostic_session,
)
from lsm.landmark_stats import FINGERS, MCP_JOINTS
from lsm.measurements import (
    Phase,
    Recording,
    Tally,
    amplitude_ratio,
    attempt_bounds,
    lag_frames,
    measure,
    percentile,
    phases_from_states,
    stroke_signals,
    window_speeds_per_s,
)
from lsm.one_euro import OneEuroFilter, OneEuroParams
from lsm.plausibility import ImplausibleKind, PlausibilityParams, filter_stream
from lsm.preprocessing import NotReconstructed, Preprocessing, reconstruct
from lsm.timing import frame_times_ms
from lsm.tracking_diagnostics import REST_PREFIX
from lsm.types import FrameStream, InvalidFrame, InvalidReason, RawFrame, SampleKind

#: Las letras del reporte y el control. Las demás se miden igual pero no se
#: imprimen.
LETRAS = ("X", "ENIE", "Q", "K")
CONTROL_ESTATICO = "A"


def recording_from_session(
    sesion: DiagnosticSession, config: Config, letras: Sequence[str]
) -> Recording:
    """Una sesión de diagnóstico como `Recording`, con la fase de cada cuadro.

    Las letras llevan el sufijo `·diag` —la X de los diagnósticos es la de la
    definición anterior al 2026-09-29— y las posturas de reposo el suyo.
    """
    registros = sesion.records
    prompts = [r.prompt for r in registros]
    fases = phases_from_states(
        [r.state for r in registros],
        [r.wall_ms for r in registros],
        [
            (a, b)
            for a, b in attempt_bounds(prompts, [r.repetition for r in registros])
            if not (prompts[a] or "").startswith(REST_PREFIX)
        ],
        config.diagnostics.pre_candidate_ms,
    )
    letras_cuadro: list[str | None] = []
    fases_cuadro: list[Phase | None] = []
    for prompt, fase in zip(prompts, fases, strict=True):
        if prompt is not None and prompt.startswith(REST_PREFIX):
            letras_cuadro.append(f"{prompt}·diag")
            fases_cuadro.append(Phase.REPOSO)
        elif prompt is not None and prompt in letras:
            letras_cuadro.append(f"{prompt}·diag")
            fases_cuadro.append(fase)
        else:
            letras_cuadro.append(None)
            fases_cuadro.append(None)
    return Recording(
        source=f"diag:{sesion.name}",
        group=f"diag:{sesion.name}",
        stream=sesion.stream,
        letters=tuple(letras_cuadro),
        phases=tuple(fases_cuadro),
        fps=float(config.capture.camera_fps),
        states=tuple(r.state for r in registros),
    )


def recordings_from_samples(
    raiz: Path,
    config: Config,
    letras: Sequence[str],
    *,
    dynamic_since: date,
    static_label: str,
) -> Iterator[Recording]:
    """Las muestras dinámicas de `letras` grabadas desde `dynamic_since`, con
    su trazo (DURANTE) y su reposo (FINAL), y las estáticas de `static_label`.
    Sufijos `·hoy` y `·dataset`."""
    for ruta in iter_sample_paths(raiz):
        muestra = read_sample(ruta)
        meta = muestra.metadata
        grupo = f"raw:{meta.signer_id}/{meta.session_id}"
        n = len(muestra.frames)
        if (
            meta.kind is SampleKind.DYNAMIC
            and meta.label in letras
            and meta.timestamp.date() >= dynamic_since
        ):
            trazo = meta.stroke_frames if meta.stroke_frames is not None else n
            yield Recording(
                source=str(ruta),
                group=grupo,
                stream=muestra.frames,
                letters=(f"{meta.label}·hoy",) * n,
                phases=tuple(
                    Phase.DURANTE if i < trazo else Phase.FINAL for i in range(n)
                ),
                fps=float(config.capture.camera_fps),
            )
        elif meta.kind is SampleKind.STATIC and meta.label == static_label:
            yield Recording(
                source=str(ruta),
                group=grupo,
                stream=muestra.frames,
                letters=(f"{meta.label}·dataset",) * n,
                phases=(Phase.ESTATICA,) * n,
                fps=float(config.capture.camera_fps),
            )


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #


def _f(value: float | None, pattern: str = "{:.2f}") -> str:
    return "—" if value is None else pattern.format(value)


def _p(values: Sequence[float], qs: Sequence[float], pattern: str = "{:.2f}") -> str:
    return " / ".join(_f(percentile(values, q), pattern) for q in qs)


def _tabla(cabeceras: Sequence[str], filas: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(cabeceras) + " |",
        "|" + "|".join("---" for _ in cabeceras) + "|",
        *("| " + " | ".join(fila) + " |" for fila in filas),
    ]


#: Cortes del acumulado de huecos, en ms.
_CORTES = tuple(range(50, 1050, 50))
_CORTES_TXT = tuple(f"≤{c}" for c in _CORTES)

#: Umbrales candidatos de la diferencia absoluta de huesos, en palmas.
_UMBRALES_HUESO = (0.2, 0.3, 0.4, 0.45, 0.5)

#: Umbrales candidatos del salto de la palma, en palmas por segundo.
_UMBRALES_SALTO = (15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0)


def _acumulado(ms: Sequence[float]) -> list[str]:
    return [f"{sum(1 for v in ms if v <= c) / len(ms):.2f}" for c in _CORTES]


def _orden(clave: str) -> tuple[int, str]:
    base = clave.split("·")[0]
    orden = [*LETRAS, CONTROL_ESTATICO]
    return (orden.index(base) if base in orden else len(orden), clave)


def render_tolerance(tally: Tally, metadata: dict[str, str]) -> str:
    partes = ["# Mediciones de tolerancia a MediaPipe (Paso 1)", ""]
    partes += [f"- **{k}**: {v}" for k, v in metadata.items()]

    letras = sorted({letra for letra, _ in tally.detection}, key=_orden)
    fases = list(Phase)

    partes += ["", "## 1. Tasa de detección por fase", ""]
    filas = []
    for letra in letras:
        celdas = [letra]
        for fase in fases:
            n, ok = tally.detection.get((letra, fase), [0, 0])
            celdas.append(f"{ok / n:.3f} ({n})" if n else "—")
        filas.append(celdas)
    partes += _tabla(["letra", *[f.value for f in fases]], filas)

    partes += [
        "",
        "## 2. Huecos por fase",
        "",
        "Del primer cuadro sin mano al primero con mano de vuelta. Duración en ms, "
        "p10 / p50 / p90 / p99; «≤ X ms» es la fracción acumulada.",
        "",
    ]
    filas = []
    for letra in letras:
        for fase in fases:
            propios = [g for g in tally.gaps if g.letter == letra and g.phase is fase]
            if not propios:
                continue
            ms = [g.duration_ms for g in propios]
            filas.append(
                [
                    letra,
                    fase.value,
                    str(len(propios)),
                    _p(ms, (0.1, 0.5, 0.9, 0.99), "{:.0f}"),
                    *_acumulado(ms),
                ]
            )
    partes += _tabla(
        ["letra", "fase", "huecos", "ms p10/p50/p90/p99", *_CORTES_TXT], filas
    )
    estados = sorted({g.state for g in tally.gaps if g.state is not None})
    if estados:
        partes += [
            "",
            "### 2.1 Por estado de la máquina al empezar el hueco",
            "",
            "Solo sesiones en vivo. «X/ENIE/Q» son las tres letras del objetivo; "
            "«todas», las del reporte.",
            "",
        ]
        filas = []
        for estado in estados:
            for nombre, filtro in (
                ("X/ENIE/Q", ("X·diag", "ENIE·diag", "Q·diag")),
                ("todas", None),
            ):
                ms = [
                    g.duration_ms
                    for g in tally.gaps
                    if g.state == estado and (filtro is None or g.letter in filtro)
                ]
                if ms:
                    filas.append([estado, nombre, str(len(ms)), *_acumulado(ms)])
        partes += _tabla(["estado", "letras", "huecos", *_CORTES_TXT], filas)

    partes += [
        "",
        "## 3. Score del detector antes de cada pérdida",
        "",
        "MediaPipe Tasks expone un solo score por mano, el de la lateralidad; "
        "el de presencia se consume dentro del grafo (`tracking_diagnostics`). "
        "p10 / p50 del score en el cuadro válido inmediatamente anterior al "
        "hueco (−1) y en los dos previos, contra todos los cuadros válidos.",
        "",
    ]
    filas = []
    for letra in letras:
        propios = [g for g in tally.gaps if g.letter == letra]
        todos = tally.scores.get(letra, [])
        celdas = [letra, str(len(propios))]
        for k in range(3):
            valores = [g.scores_before[k] for g in propios if len(g.scores_before) > k]
            celdas.append(_p(valores, (0.1, 0.5), "{:.3f}"))
        celdas.append(_p(todos, (0.1, 0.5), "{:.3f}"))
        filas.append(celdas)
    partes += _tabla(
        ["letra", "pérdidas", "−1", "−2", "−3", "todos los válidos"], filas
    )

    partes += [
        "",
        "## 4. Largo de los huesos",
        "",
        "|L / ref − 1| de cada uno de los 20 huesos, en unidades de palma. "
        "**sesión**: ref = mediana del hueso en toda la sesión; **móvil**: ref = "
        "mediana de los últimos cuadros válidos (la del filtro). «hueso» = p50 / "
        "p95 sobre todos los huesos y cuadros; «peor hueso» = p50 / p95 / p99 del "
        "máximo por cuadro (lo que decide un filtro por cuadro).",
        "",
    ]
    filas = []
    for letra in letras:
        for dim in ("2d", "3d"):
            for ref in ("sesion", "movil"):
                todos = tally.bone_dev.get((letra, dim, ref), [])
                peores = tally.bone_dev_max.get((letra, dim, ref), [])
                if not todos:
                    continue
                filas.append(
                    [
                        letra,
                        dim,
                        ref,
                        _p(todos, (0.5, 0.95), "{:.3f}"),
                        _p(peores, (0.5, 0.95, 0.99), "{:.3f}"),
                        str(len(peores)),
                    ]
                )
    partes += _tabla(
        ["letra", "dim", "ref", "hueso p50/p95", "peor hueso p50/p95/p99", "cuadros"],
        filas,
    )
    partes += [
        "",
        "### 4.1 Diferencia absoluta contra la mediana móvil",
        "",
        "max_hueso |L − ref| por cuadro, en palmas: la medida del filtro de "
        "plausibilidad (los huesos cortos no dominan, como en la relativa). "
        "p50 / p95 / p99 / p99.9, y la fracción de cuadros que invalidaría cada "
        "umbral.",
        "",
    ]
    filas = []
    for letra in letras:
        for dim in ("2d", "3d"):
            peores = tally.bone_abs_max.get((letra, dim), [])
            if not peores:
                continue
            filas.append(
                [
                    letra,
                    dim,
                    _p(peores, (0.5, 0.95, 0.99, 0.999), "{:.3f}"),
                    *(
                        f"{sum(1 for v in peores if v > u) / len(peores):.3f}"
                        for u in _UMBRALES_HUESO
                    ),
                ]
            )
    partes += _tabla(
        [
            "letra",
            "dim",
            "p50/p95/p99/p99.9",
            *(f"> {u:g}" for u in _UMBRALES_HUESO),
        ],
        filas,
    )

    partes += [
        "",
        "## 5. Elevación dorsal de la falange proximal",
        "",
        "`landmark_stats.mcp_dorsal_elevation`, grados: negativa doblando hacia "
        "la palma, positiva hacia el dorso. p0.5 / p99.5 / max por dedo.",
        "",
    ]
    filas = []
    for letra in letras:
        celdas = [letra]
        for dedo, *_ in MCP_JOINTS:
            valores = tally.mcp_dorsal.get((letra, dedo), [])
            celdas.append(
                _p(valores, (0.005, 0.995), "{:.0f}")
                + (f" / {max(valores):.0f}" if valores else "")
            )
        filas.append(celdas)
    partes += _tabla(["letra", *(d for d, *_ in MCP_JOINTS)], filas)

    partes += [
        "",
        "## 6. Saltos de la palma",
        "",
        "Desplazamiento del centro de la palma (muñeca y nudillos) entre cuadros "
        "válidos, en palmas por segundo de reloj real. p50 / p99 / p99.9 / max.",
        "",
    ]
    filas = []
    for letra in letras:
        seguidos = tally.jumps.get(letra, [])
        tras_hueco = tally.jumps_after_gap.get(letra, [])
        if not seguidos and not tras_hueco:
            continue
        filas.append(
            [
                letra,
                _p(seguidos, (0.5, 0.99, 0.999), "{:.2f}")
                + (f" / {max(seguidos):.2f}" if seguidos else ""),
                _p(tras_hueco, (0.5, 0.99), "{:.2f}")
                + (f" / {max(tras_hueco):.2f}" if tras_hueco else ""),
            ]
        )
    partes += _tabla(["letra", "cuadros seguidos", "tras un hueco"], filas)
    partes += [
        "",
        "Fracción de saltos entre cuadros seguidos por encima de cada umbral:",
        "",
    ]
    filas = [
        [
            letra,
            *(
                f"{sum(1 for v in saltos if v > u) / len(saltos):.4f}"
                for u in _UMBRALES_SALTO
            ),
        ]
        for letra in letras
        if (saltos := tally.jumps.get(letra, []))
    ]
    partes += _tabla(["letra", *(f"> {u:g}" for u in _UMBRALES_SALTO)], filas)

    posturas = sorted({p for p, _ in tally.tremor})
    if posturas:
        partes += [
            "",
            "## 7. Temblor en reposo, por dedo",
            "",
            "Velocidad de los puntos de cada dedo relativa a la muñeca, contra el "
            "cuadro de hace la ventana de reposo, en palmas por segundo; «palma» es "
            "el centro de la palma sin restar nada. p50 / p95.",
            "",
        ]
        filas = []
        for postura in posturas:
            filas.append(
                [
                    postura,
                    *(
                        _p(tally.tremor.get((postura, dedo), []), (0.5, 0.95))
                        for dedo in FINGERS
                    ),
                    _p(tally.tremor_palm.get(postura, []), (0.5, 0.95)),
                ]
            )
        partes += _tabla(["postura", *FINGERS, "palma"], filas)
    return "\n".join(partes) + "\n"


def _cmd_tolerancia(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tally = Tally()
    sesiones = 0
    # Sesión a sesión: cada una es su propio grupo de medianas, y tenerlas
    # todas en memoria a la vez son cientos de miles de cuadros.
    for carpeta in iter_diagnostic_folders(args.diagnosticos):
        sesion = load_diagnostic_session(carpeta)
        medir(
            [recording_from_session(sesion, config, LETRAS)],
            tally,
            config,
            args.cuadros_moviles,
        )
        sesiones += 1
    muestras = list(
        recordings_from_samples(
            args.raiz,
            config,
            LETRAS,
            dynamic_since=args.desde,
            static_label=CONTROL_ESTATICO,
        )
    )
    medir(muestras, tally, config, args.cuadros_moviles)
    texto = render_tolerance(
        tally,
        {
            "sesiones de diagnóstico": str(sesiones),
            "muestras (dinámicas desde "
            f"{args.desde.isoformat()} y {CONTROL_ESTATICO} estáticas)": str(
                len(muestras)
            ),
            "ventana de la mediana móvil": f"{args.cuadros_moviles} cuadros",
            "ventana del temblor": f"{config.diagnostics.rest_windows_ms[0]:g} ms",
            "sufijos": "·diag = sesiones de diagnóstico (la X ahí es la de la "
            "definición anterior); ·hoy = muestras dinámicas desde la fecha; "
            "·dataset = muestras estáticas",
        },
    )
    _escribir(texto, args.salida)
    return 0


def contar_invalidados(
    stream: FrameStream,
    letras: Sequence[str | None],
    params: PlausibilityParams,
    fps: float,
    cuenta: dict[str, list[int]],
) -> None:
    """Suma, por letra, [cuadros con mano, BONE, JOINT, JUMP] del flujo."""
    for slot, original, letra in zip(
        filter_stream(stream, params, fps), stream, letras, strict=True
    ):
        if letra is None or not isinstance(original, RawFrame):
            continue
        fila = cuenta.setdefault(letra, [0, 0, 0, 0])
        fila[0] += 1
        if isinstance(slot, InvalidFrame) and slot.reason is InvalidReason.IMPLAUSIBLE:
            fila[1 + list(ImplausibleKind).index(ImplausibleKind(slot.detail))] += 1


def render_plausibility(
    cuenta: dict[str, list[int]], metadata: dict[str, str], limite: float
) -> str:
    partes = ["# Fracción de cuadros invalidados por plausibilidad (Paso 2)", ""]
    partes += [f"- **{k}**: {v}" for k, v in metadata.items()]
    filas = []
    for letra in sorted(cuenta):
        n, hueso, articulacion, salto = cuenta[letra]
        fraccion = (hueso + articulacion + salto) / n if n else 0.0
        aviso = " ⚠" if fraccion > limite else ""
        filas.append(
            [
                letra,
                str(n),
                f"{fraccion:.4f}{aviso}",
                f"{hueso / n:.4f}" if n else "—",
                f"{articulacion / n:.4f}" if n else "—",
                f"{salto / n:.4f}" if n else "—",
            ]
        )
    partes += [
        "",
        f"⚠ marca las letras por encima de {limite:.0%}.",
        "",
        *_tabla(
            [
                "letra",
                "cuadros con mano",
                "invalidados",
                "hueso",
                "articulación",
                "salto",
            ],
            filas,
        ),
    ]
    return "\n".join(partes) + "\n"


def _cmd_plausibilidad(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    params = PlausibilityParams.from_config(config)
    fps = float(config.capture.camera_fps)
    cuenta: dict[str, list[int]] = {}
    for carpeta in iter_diagnostic_folders(args.diagnosticos):
        sesion = load_diagnostic_session(carpeta)
        contar_invalidados(
            sesion.stream,
            [None if r.prompt is None else f"{r.prompt}·diag" for r in sesion.records],
            params,
            fps,
            cuenta,
        )
    muestras = 0
    for ruta in iter_sample_paths(args.raiz):
        muestra = read_sample(ruta)
        meta = muestra.metadata
        sufijo = "est" if meta.kind is SampleKind.STATIC else "din"
        contar_invalidados(
            muestra.frames,
            [f"{meta.label}·{sufijo}"] * len(muestra.frames),
            params,
            fps,
            cuenta,
        )
        muestras += 1
    texto = render_plausibility(
        cuenta,
        {
            "sesiones de diagnóstico": args.diagnosticos.as_posix(),
            "muestras del dataset": f"{muestras} ({args.raiz.as_posix()})",
            "plausibility": ", ".join(
                f"{k}={v}" for k, v in config.plausibility.model_dump().items()
            ),
            "sufijos": "·diag = diagnóstico en vivo (cada muestra y cada sesión "
            "es un flujo aparte: la referencia empieza vacía); ·est / ·din = "
            "muestras estáticas / dinámicas del dataset, todas las fechas",
        },
        args.limite,
    )
    _escribir(texto, args.salida)
    return 0


#: Rejilla del barrido del One Euro: min_cutoff (Hz) × beta (1/palma).
_MIN_CUTOFFS = (0.25, 0.5, 1.0, 2.0, 4.0)
_BETAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
#: Letras cuyos trazos se miden: la J es la referencia de τ; Ñ y Q, del giro.
_LETRAS_TRAZO = ("J", "ENIE", "Q")


def _reposos(
    raiz: Path, config: Config, plaus: PlausibilityParams
) -> dict[str, list[tuple[list[RawFrame], list[float]]]]:
    """Tramos válidos de cada postura de reposo, sin el asentamiento."""
    tramos: dict[str, list[tuple[list[RawFrame], list[float]]]] = {}
    for carpeta in iter_diagnostic_folders(raiz):
        sesion = load_diagnostic_session(carpeta)
        if not any((r.prompt or "").startswith(REST_PREFIX) for r in sesion.records):
            continue
        flujo = tuple(filter_stream(sesion.stream, plaus, 30.0))
        actual: list[RawFrame] = []
        tiempos: list[float] = []
        postura: str | None = None
        inicio = 0.0
        for slot, r in zip(flujo, sesion.records, strict=True):
            pose = r.prompt if (r.prompt or "").startswith(REST_PREFIX) else None
            if pose != postura:
                if postura and len(actual) > 5:
                    tramos.setdefault(postura, []).append((actual, tiempos))
                actual, tiempos, postura, inicio = [], [], pose, r.wall_ms
            if (
                postura is None
                or r.wall_ms - inicio < config.diagnostics.rest_settle_ms
            ):
                continue
            if isinstance(slot, RawFrame):
                actual.append(slot)
                tiempos.append(r.wall_ms)
        if postura and len(actual) > 5:
            tramos.setdefault(postura, []).append((actual, tiempos))
    return tramos


def _trazos(
    raiz: Path, config: Config, desde: datetime
) -> dict[str, list[tuple[list[RawFrame], list[float]]]]:
    """El trazo de cada muestra dinámica de J, Ñ y Q grabada desde `desde`,
    reconstruido sin One Euro (plausibilidad y huecos), a la tasa nominal."""
    pre = Preprocessing.from_config(config, float(config.capture.camera_fps))
    pre = replace(pre, one_euro=replace(pre.one_euro, enabled=False))
    trazos: dict[str, list[tuple[list[RawFrame], list[float]]]] = {}
    for ruta in iter_sample_paths(raiz):
        muestra = read_sample(ruta)
        meta = muestra.metadata
        if (
            meta.kind is not SampleKind.DYNAMIC
            or meta.label not in _LETRAS_TRAZO
            or meta.timestamp < desde
        ):
            continue
        flujo = muestra.frames[: meta.stroke_frames or len(muestra.frames)]
        hecha = reconstruct(flujo, SampleKind.DYNAMIC, pre)
        if isinstance(hecha, NotReconstructed) or len(hecha.sequence) < 12:
            continue
        frames = list(hecha.sequence.frames)
        trazos.setdefault(meta.label, []).append(
            (frames, list(frame_times_ms(tuple(frames), pre.fps)))
        )
    return trazos


def _filtrar(
    frames: list[RawFrame], tiempos: list[float], params: OneEuroParams
) -> list[RawFrame]:
    filtro = OneEuroFilter(params)
    return [filtro.step(f, t) for f, t in zip(frames, tiempos, strict=True)]


def _cmd_one_euro(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plaus = PlausibilityParams.from_config(config)
    reposos = _reposos(args.diagnosticos, config, plaus)
    trazos = _trazos(args.raiz, config, args.desde)
    ventana = config.segmentation.velocity_window_ms
    umbral_quieto = config.segmentation.velocity_threshold_per_s
    d_cutoff = args.d_cutoff

    def reposo(params: OneEuroParams) -> tuple[float, float]:
        """p95 de la velocidad de ventana y fracción ≥ velocity_threshold."""
        velocidades: list[float] = []
        for tramos in reposos.values():
            for frames, tiempos in tramos:
                filtrados = _filtrar(frames, tiempos, params)
                velocidades += window_speeds_per_s(filtrados, tiempos, ventana)
        p95 = percentile(velocidades, 0.95) or 0.0
        sobre = sum(1 for v in velocidades if v >= umbral_quieto) / max(
            1, len(velocidades)
        )
        return p95, sobre

    def trazo(params: OneEuroParams) -> dict[str, tuple[float, float, float]]:
        """Por letra: retraso mediano de τ y del giro, en ms, y amplitud del giro."""
        salida: dict[str, tuple[float, float, float]] = {}
        for letra, lista in trazos.items():
            lag_tau: list[float] = []
            lag_giro: list[float] = []
            amplitud: list[float] = []
            for frames, tiempos in lista:
                paso = (tiempos[-1] - tiempos[0]) / (len(tiempos) - 1)
                crudas = stroke_signals(frames)
                filtradas = stroke_signals(_filtrar(frames, tiempos, params))
                for eje in ("tau_x", "tau_y"):
                    if amplitude_ratio(crudas[eje], filtradas[eje]) is not None:
                        lag_tau.append(
                            lag_frames(crudas[eje], filtradas[eje], 8) * paso
                        )
                lag_giro.append(lag_frames(crudas["giro"], filtradas["giro"], 8) * paso)
                a = amplitude_ratio(crudas["giro"], filtradas["giro"])
                if a is not None:
                    amplitud.append(a)
            salida[letra] = (
                percentile(lag_tau, 0.5) or 0.0,
                percentile(lag_giro, 0.5) or 0.0,
                percentile(amplitud, 0.5) or 0.0,
            )
        return salida

    base = OneEuroParams(enabled=False, min_cutoff=1.0, beta=0.0, d_cutoff=d_cutoff)
    p95_crudo, sobre_crudo = reposo(base)
    filas: list[list[str]] = []
    for min_cutoff in _MIN_CUTOFFS:
        for beta in _BETAS:
            params = OneEuroParams(
                enabled=True, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
            )
            p95, sobre = reposo(params)
            por_letra = trazo(params)
            filas.append(
                [
                    f"{min_cutoff:g}",
                    f"{beta:g}",
                    f"{p95:.3f} ({p95 / p95_crudo - 1:+.0%})",
                    f"{sobre:.3f}",
                    *(
                        f"{por_letra[letra][0]:.0f} / {por_letra[letra][1]:.0f} / "
                        f"{por_letra[letra][2]:.2f}"
                        if letra in por_letra
                        else "—"
                        for letra in _LETRAS_TRAZO
                    ),
                ]
            )
    partes = [
        "# Calibración del One Euro (Paso 3)",
        "",
        f"- **reposo**: {sum(len(v) for v in reposos.values())} tramos de "
        f"{', '.join(sorted(reposos))} (diagnósticos, reloj real, sin los "
        f"{config.diagnostics.rest_settle_ms:g} ms de asentamiento)",
        "- **trazos**: "
        + ", ".join(f"{letra} {len(v)}" for letra, v in sorted(trazos.items()))
        + f" (muestras dinámicas desde {args.desde.isoformat()}, reloj nominal)",
        f"- **d_cutoff**: {d_cutoff:g} Hz",
        f"- **sin filtro**: velocidad de ventana en reposo p95 {p95_crudo:.3f} "
        f"palmas/s, fracción ≥ velocity_threshold ({umbral_quieto:g}) "
        f"{sobre_crudo:.3f}",
        "",
        "Reposo: p95 de la velocidad con la que decide la máquina (ventana de "
        f"{ventana:g} ms) y fracción de cuadros que la máquina llamaría "
        "movimiento. Trazos, por letra: retraso mediano de τ / del giro en ms y "
        "amplitud mediana del giro (1 = intacta).",
        "",
        *_tabla(
            ["min_cutoff", "beta", "reposo p95", "≥ reposo", *_LETRAS_TRAZO], filas
        ),
    ]
    _escribir("\n".join(partes) + "\n", args.salida)
    return 0


def medir(
    grabaciones: list[Recording],
    tally: Tally,
    config: Config,
    cuadros_moviles: int = 15,
) -> Tally:
    return measure(
        grabaciones,
        tally,
        rolling_frames=cuadros_moviles,
        rest_window_ms=config.diagnostics.rest_windows_ms[0],
        rest_settle_ms=config.diagnostics.rest_settle_ms,
    )


def _escribir(texto: str, salida: Path | None) -> None:
    print(texto)
    if salida is not None:
        salida.parent.mkdir(parents=True, exist_ok=True)
        salida.write_text(texto, encoding="utf-8")
        print(f"reporte escrito en {salida}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsm-medir",
        description="Mediciones de la tolerancia a las fallas de MediaPipe.",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--raiz", type=Path, default=Path("data/raw"))
    parser.add_argument("--diagnosticos", type=Path, default=Path("data/diagnostico"))
    parser.add_argument("--salida", type=Path, default=None)
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    tolerancia = subcomandos.add_parser(
        "tolerancia",
        help="Paso 1: huecos, scores, huesos, ángulos, saltos y temblor",
    )
    tolerancia.add_argument(
        "--desde",
        type=date.fromisoformat,
        default=date(2026, 9, 29),
        help="fecha desde la que se toman las muestras dinámicas",
    )
    tolerancia.add_argument(
        "--cuadros-moviles",
        type=int,
        default=15,
        dest="cuadros_moviles",
        help="cuadros de la mediana móvil de los huesos",
    )
    tolerancia.set_defaults(func=_cmd_tolerancia)
    plausibilidad = subcomandos.add_parser(
        "plausibilidad",
        help="Paso 2: fracción de cuadros que invalida el filtro, por letra",
    )
    plausibilidad.add_argument(
        "--limite",
        type=float,
        default=0.10,
        help="fracción por encima de la cual se marca la letra (por defecto 0.10)",
    )
    plausibilidad.set_defaults(func=_cmd_plausibilidad)
    one_euro = subcomandos.add_parser(
        "one-euro",
        help="Paso 3: barrido del One Euro, temblor en reposo contra retraso",
    )
    one_euro.add_argument(
        "--desde",
        type=datetime.fromisoformat,
        default=datetime.fromisoformat("2026-09-29T00:00:00-06:00"),
        help="instante desde el que se toman los trazos (ISO-8601 con zona)",
    )
    one_euro.add_argument(
        "--d-cutoff", type=float, default=1.0, dest="d_cutoff", help="en Hz"
    )
    one_euro.set_defaults(func=_cmd_one_euro)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
