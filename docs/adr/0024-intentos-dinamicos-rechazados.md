# ADR 0024 — Los intentos dinámicos rechazados se guardan (Paso 0)

- **Estado:** **implementada** el 2026-09-29.
- **Fecha:** 2026-09-29
- **Fase:** 5.1, serie de tolerancia a las fallas de MediaPipe (Paso 0)
- **Relacionadas:** ADR 0021 (huecos), ADR 0023 (captura delimitada por la
  máquina)

## Contexto

Desde el ADR 0023 la captura de una dinámica la delimita la máquina de estados, y
una grabación que no entrega un trazo cerrado en `capture.dynamic_max_ms` se
rechaza **sin guardar nada**. En X, Ñ y Q eso es lo más frecuente: el trazo se
corta por un hueco largo y el tope llega sin trazo. La serie de cambios que abre
este ADR va a hacer la tubería más tolerante a esos huecos; cada intento
rechazado hoy es uno que mañana podría servir, y se estaba tirando.

## Decisión

1. **Todo rechazo de una dinámica guarda el intento entero**, en crudo: lo que la
   cámara entregó desde que ESPACIO armó la grabación hasta el rechazo, huecos
   incluidos, con el motivo, la tasa con la que la máquina convirtió sus umbrales
   y la sección `segmentation` de la configuración. Formato y ubicación en
   `docs/dataset-schema.md` («Los intentos dinámicos rechazados»).
2. **Motivos más finos** (`lsm.capture.Rejection`). El tope de tiempo ya no dice
   siempre «no se cerró el trazo»: si lo último que le pasó a un candidato fue un
   hueco que no se pudo rellenar, es `STROKE_INTERRUPTED`; si se cerró demasiado
   reconstruido, `TOO_MUCH_INTERPOLATED`. `evaluate_window` separa también el
   exceso de relleno (`TOO_MUCH_INTERPOLATED`) del hueco largo (`HAS_GAPS`).
3. **Fuera del entrenamiento por construcción.** Viven en `<LETRA>/rechazados/`,
   un nivel por debajo de las muestras: `iter_sample_paths` no los alcanza (y los
   excluye además por nombre), `read_sample` rechaza su esquema y el contador del
   preview no los cuenta. No hace falta ningún filtro que alguien pueda olvidar.
4. **Se re-segmentan con la captura misma.** `lsm.capture.resegment_attempt`
   pasa el intento por `delimit_dynamic_stroke` y `evaluate_window`, que es lo que
   hace `lsm-capture grabar`, con la tasa guardada. `lsm-capture rechazados`
   cuenta cuántos darían hoy un trazo aceptable, por letra y motivo, sin escribir
   nada.

## Lo que no hace

- **No recupera muestras.** Convertir un intento rescatado en muestra —con qué
  procedencia, y si se mezcla con las grabadas— se decide cuando haya intentos
  guardados que rescatar. Hoy no hay ninguno: el mecanismo empieza con la
  siguiente sesión.
- **No guarda las estáticas rechazadas.** Una estática rechazada es una ventana
  que se volvió a sostener un segundo después; no se pierde nada.
