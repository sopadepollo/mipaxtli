# ADR 0030 — Bajar los umbrales de MediaPipe con la red puesta (Paso 5)

- **Estado:** **pendiente de medir en vivo.** Las herramientas están; el barrido
  necesita la cámara: sin video guardado no hay forma de volver a pasar
  MediaPipe con otros umbrales sobre las sesiones grabadas.
- **Fecha:** 2026-09-29
- **Relacionadas:** ADR 0017 (diagnóstico de tracking), ADR 0026 a 0029

## Contexto

`hands.min_hand_presence_confidence` y `hands.min_tracking_confidence` están en
0.5, el valor por defecto de MediaPipe. Bajarlos da más continuidad a costa de
aceptar manos peores. Con la plausibilidad (ADR 0027) y el relleno ampliado (ADR
0029) activos hay red para lo segundo: un cuadro malo que pase el detector puede
invalidarse y rellenarse.

El ADR 0026 midió que el score del detector cae antes de una parte de las
pérdidas (p10 del cuadro previo a un hueco: 0.68 en la X contra 0.92 en todos
sus cuadros), pero no en la mitad: bajar el umbral puede recuperar las primeras,
no las pérdidas repentinas del rastreador.

## Qué se mide

Por cada par de valores, una sesión `lsm-demo diagnosticar` con las mismas
letras y la misma luz, y `lsm-medir mediapipe` las agrupa por par. Para X, Ñ y Q:

- **detección**: fracción de cuadros con mano durante el trazo;
- **implausibles**: de esos, los que invalida la plausibilidad;
- **enteros**: intentos con exactamente un trazo entregado (con la máquina de la
  sesión: la del Paso 4).

**El valor bueno sube la detección sin que los implausibles suban mucho.**

## Línea base (las 10 sesiones guiadas existentes, 0.5 / 0.5)

| letra | detección | implausibles | enteros |
|---|---|---|---|
| Ñ | 0.780 | 0.0056 | 40 / 101 |
| Q | 0.579 | 0.0107 | 6 / 111 |
| X | 0.660 | 0.0157 | 6 / 93 |

Grabadas antes de los pasos 2 a 4 (y la X con la definición anterior): sus
«enteros» son de la máquina de entonces. Los de las sesiones nuevas no son
comparables uno a uno; la comparación válida es entre sesiones del barrido.

## Plan

Rejilla mínima, de más conservador a menos: presencia × tracking en
{0.5, 0.4, 0.3} × {0.5, 0.35, 0.2}, empezando por la diagonal (0.4/0.35,
0.3/0.2). Una sesión por punto, `--repeticiones 5` para que quepa en una tarde.
Los comandos están en `docs/COMO-PROBAR.md` §6.2.
