# ADR 0023 — La captura de dinámicas la delimita la máquina de estados (Bloque 4)

- **Estado:** **implementada** el 2026-09-28. Falta grabar las dinámicas nuevas y
  reconstruir las plantillas.
- **Fecha:** 2026-09-28
- **Fase:** 5.1, Bloque 4
- **Relacionadas:** ADR 0007 (cierre de captura), ADR 0015 (camino dinámico), ADR
  0017 (diagnóstico), ADR 0021 (huecos)

## Contexto

Hasta aquí una dinámica se grababa a mano: ESPACIO empezaba y ESPACIO cerraba, y
un tope en cuadros (`capture.dynamic_max_frames`, 90 = 3 s) **congelaba** la
grabación si nadie la cerraba. En vivo, en cambio, lo que llega al clasificador
es el trazo que delimita la máquina de estados. Las dos cosas no coincidían:

- **Muchas muestras están truncadas.** 529 de las 822 dinámicas grabadas acaban
  con la mano todavía en movimiento (velocidad de cierre del último frame ≥
  `motion_threshold_per_s`): K 99 de 121, X 81 de 101, Z 74, Q 72, J 64, Ñ 62, LL
  42 y RR 35 de 100.
- **La demo lo tapaba.** Al reproducir el dataset fabricaba un reposo final
  repitiendo el último frame; sin él los trazos nunca se cerraban.
- Las plantillas del DTW salen de esas muestras, y el ADR 0022 encontró que el
  DTW rechaza por margen trazos en vivo que acierta.

## Decisión

1. **La muestra dinámica la delimita la misma máquina que usa la demo.** ESPACIO
   **arma** la grabación; la captura pasa la cámara por `run_segmentation`, con
   la tasa medida y un clasificador que no reconoce nada, y la muestra es el
   trazo que la máquina entregaría —desde su frame de partida— **más el reposo
   que lo cerró**, hasta DYNAMIC_EMIT (`lsm.capture.delimit_dynamic_stroke`).
   `WindowDynamic` gana `start_frame_index` para eso.
   - El plan decía «empieza al entrar en DYNAMIC_CANDIDATE». No coincide con lo
     que recibe el clasificador: el candidato nace tras `motion_min_ms` de
     movimiento, pero el trazo que se clasifica empieza en el frame anterior al
     primer movimiento. Se guarda desde ahí, para que el modelo entrene con los
     mismos segmentos que recibe.
   - **Se entrena solo el trazo**: la muestra anota `stroke_frames` (esquema v4)
     y `to_sample` corta ahí. El reposo queda en el archivo para que reproducirla
     cierre el trazo sola.
2. **Tope en tiempo, y el que se pasa se rechaza.** `capture.dynamic_max_ms` =
   6000 reemplaza a `dynamic_max_frames`. Si en ese tiempo la máquina no entrega
   un trazo cerrado, no se guarda nada (`Rejection.STROKE_NOT_CLOSED`, «no se
   cerro el trazo: deten la mano al terminar»); si lo descarta por pasar de
   `motion_max_ms`, tampoco (`STROKE_TOO_LONG`). Un candidato interrumpido por un
   hueco largo no rechaza: la grabación sigue y se puede repetir la letra antes
   del tope.
3. **Las truncadas se marcan, no se borran.** `lsm-capture marcar-truncadas`
   escribe `data/raw/truncadas.json` con las dinámicas sin `stroke_frames` cuya
   velocidad de cierre final supera `motion_threshold_per_s`, y `load_corpus` las
   deja fuera (anotado en la procedencia, `excluded_truncated`). Borrar el
   manifiesto las devuelve. Se aconseja marcarlas **después** de grabar las
   nuevas: con las de hoy quedarían entre 20 y 38 por letra.
4. **La demo ya no fabrica el reposo** (`flujo_desde_dataset`). Una muestra nueva
   se cierra sola; una vieja truncada sale interrumpida, que es lo que es.
   `evaluation.replay_sample` sí sigue añadiendo reposo: reproduce
   `Sample.sequence`, que es el trazo sin el reposo por construcción.

### Consecuencia en `lsm-capture verificar`

Desde `FEATURE_SPEC_VERSION` 3 (ADR 0020) la σ re-derivada de una muestra vieja
no coincide con la que anotó: la escala del paso 4 cambió. Las muestras v4 anotan
con qué versión calcularon su σ (`feature_spec_version`), y `verificar` solo la
compara cuando coincide con la actual; de las demás revisa huecos y extracción y
las cuenta aparte. Las 3407 de hoy caen en ese grupo.

## Verificación

- Tests: una mano que se mueve y se detiene da el trazo desde su frame de partida,
  con el reposo; si el flujo acaba con la mano en marcha, `STROKE_NOT_CLOSED`; un
  trazo interminable, `STROKE_TOO_LONG`; una muestra con `stroke_frames` entrena
  solo el trazo y conserva el reposo en disco; el manifiesto excluye sin borrar.
- En una copia de 40 dinámicas del dataset, `marcar-truncadas` marca 34 y el
  corpus carga 6.
- El bucle de la cámara no tiene test: es el envoltorio. La decisión —dónde
  empieza y termina el trazo— es `delimit_dynamic_stroke`, que es pura.

## Lo que falta

- **Grabar** las dinámicas nuevas y reentrenar. La X queda fuera hasta resolver en
  el glosario cómo se ejecuta: hoy mide 0.17 de escala/palma contra 0.73 en el
  dataset (ADR 0020). LL y RR, al final.
- **Marcar** las truncadas una vez haya dinámicas nuevas.
