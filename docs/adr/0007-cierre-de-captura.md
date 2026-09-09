# ADR 0007 — Cierre de la captura: convención de lateralidad, calidad dinámica y bloqueo de sesión

- **Estado:** aceptada
- **Fecha:** 2026-09-08
- **Fase:** 1.5 (cierre de la Fase 1, antes de la primera sesión formal)
- **Implementa:** `src/lsm/types.py` · `src/lsm/capture.py` ·
  `src/lsm/io/calibration.py` · `src/lsm/io/glossary.py` ·
  `src/lsm/classifiers/base.py` · `src/lsm/cli/capture.py`
- **Continúa:** `docs/adr/0006-deteccion-de-manos-y-captura.md`

## Contexto

La Fase 1 dejó la captura funcionando y tres agujeros que solo se ven cuando uno
se pregunta qué pasa si algo sale mal **sin dar ningún síntoma**. Los tres tienen
la misma forma: producen un dataset de aspecto impecable, con métricas normales,
que solo se revela roto mucho después y cuya reparación es volver a citar a todo
el mundo.

Esta ADR los cierra antes de la primera sesión formal, que es el último momento en
que cerrarlos es barato.

---

## Decisión 1 — La convención de lateralidad viaja con los datos y con el modelo

### El problema, dicho con precisión

`hands.mediapipe_reports_mirrored_handedness` decide si se invierte la lateralidad
que reporta el detector. Si está al revés, el paso 2 de `feature-spec.md` canoniza
**todas** las muestras hacia la mano contraria.

**Y no pasa nada.** Las dos poblaciones de vectores difieren por un espejo global y
por nada más: el modelo entrena igual de bien, infiere igual de bien, y la matriz
de confusión sale idéntica. No hay ninguna métrica que se degrade, ningún test que
se ponga rojo, ningún síntoma que mirar.

El error tiene exactamente un escenario en el que duele: **cuando dos
implementaciones usan convenciones distintas.** Es el escenario de la Fase 7, con
MediaPipe JS trayendo la suya.

Y lo que lo vuelve peligroso es que la defensa que el proyecto ya tenía **no lo
cubre**: los golden vectors reciben la lateralidad ya resuelta como entrada, así
que `web/tests/features.test.ts` pasaría en verde mientras la app web confunde cada
seña con su espejo.

### Lo que se hace

**Un tipo, no un comentario.** `types.HandednessConvention` con dos valores —
`SIGNER` (la mano anatómica de quien firma, la del proyecto) e `IMAGE` (la que se
ve en la imagen espejada, que es como MediaPipe decide) — y la constante
`HANDEDNESS_CONVENTION`.

**Cada muestra registra el valor efectivo del interruptor**
(`handedness_swapped`) además de la convención resultante. El campo del
interruptor es el que permite reparar: si alguien lo cambia a mitad del proyecto,
sin él el dataset queda mezclado —media parte canonizada hacia una mano, media
hacia la otra, todo con la misma pinta— y no hay forma de saber qué muestra es
cuál. Con él, la reparación es un filtro y un espejo.

**El modelo exportado lleva `handedness_convention` y el runtime rechaza la carga
si no coincide** (`ARQUITECTURA.md` §4.6). Mismo patrón que `feature_spec_version`,
mismo motivo: convertir una discrepancia silenciosa en un fallo ruidoso. **Este es
el mecanismo que protege la Fase 7**, y es el único que hay.

**`lsm-capture calibrar`** pide levantar la mano derecha, muestra la lateralidad
resuelta, exige confirmación explícita y escribe un registro con identificador de
cámara, resolución, fecha y el valor del interruptor. Es el único punto del
proyecto donde una persona aporta información que ninguna prueba puede producir:
el sistema no tiene forma de saber qué mano hay delante.

**`lsm-capture grabar` se niega a abrir la cámara sin calibración vigente.** Un
rechazo, no un aviso. Un aviso en una terminal a las nueve de la mañana, antes de
cuarenta minutos de grabación con otra persona esperando, no lo lee nadie.

### Qué significa «vigente»

Que exista un registro **para esa cámara** y **con el ajuste que está puesto
ahora**. Cambiar el interruptor en `config.yaml` invalida las calibraciones
anteriores, que es exactamente lo que debe pasar: el registro dice que alguien
miró la pantalla y vio `RIGHT` con *aquel* ajuste; con el contrario habría visto
`LEFT`.

**No hay caducidad por tiempo.** Una calibración no se estropea sola: se estropea
al cambiar la configuración o la cámara, y las dos cosas ya se detectan. Añadir una
fecha de expiración sería un umbral arbitrario y una molestia periódica que no tapa
ningún fallo real.

La identidad de la cámara es `backend:índice@anchoxalto`. No hay forma portable de
leer el número de serie de una webcam, así que se compone con lo que sí cambia el
resultado: el índice distingue la integrada de la externa, la resolución **realmente
entregada** cambia el recorte y la relación de aspecto, y el backend (`V4L2`,
`DSHOW`, `AVFOUNDATION`) cambia el comportamiento del mismo dispositivo. Es una
heurística: puede confundir dos webcams idénticas conectadas en el mismo orden, y
el coste de ese falso positivo es una calibración de más.

---

## Decisión 2 — Las dinámicas también tienen criterio de calidad, y es el opuesto

Hasta ahora σ solo controlaba las estáticas. Correcto —en una J el movimiento *es*
la seña— pero dejaba a `J`, `K`, `ENIE`, `Q`, `X`, `Z`, `DOBLE_L` y `DOBLE_R`
grabándose **sin ningún filtro**.

Se podía registrar una "J" en la que la persona apenas movió la mano. En el dataset
esa muestra es indistinguible de una "I": misma configuración de dedos, trayectoria
plana, etiqueta distinta. Envenena al clasificador dinámico por el lado contrario
al que tapa la clase negativa, y **no se detecta después mirando el archivo**.

**El criterio es la longitud de arco de τ** (`feature-spec.md` §3.1), con umbral
`capture.min_trajectory_arc`. Los dos modos quedan simétricos y opuestos:

| Modo | Se rechaza si | Umbral |
|---|---|---|
| `STATIC` | se movió **de más** | `quality.max_dispersion` |
| `DYNAMIC` | se movió **de menos** | `capture.min_trajectory_arc` |

Los dos números se calculan y se guardan siempre; lo que cambia es cuál decide.

### Arco y no desplazamiento neto

Es la decisión que hace que la medida funcione. La `Ñ` y la `Q` son rotaciones de
ida y vuelta; la `Z` y la `X` vuelven cerca de donde empezaron. Para todas ellas el
desplazamiento neto es casi cero mientras el recorrido es largo, así que medir el
neto rechazaría justo las letras que hay que aceptar.

El precio es que **el arco acumula el temblor del detector**: una mano quieta
durante noventa frames suma noventa pequeñas sacudidas y no da cero. Por eso el
umbral tiene que dejar margen sobre ese ruido de fondo, y por eso se calibra con
trazos reales y no a ojo.

### Los valores son puntos de partida, no medidas

`min_trajectory_arc = 0.8` unidades de mano. Razonado así: el gancho de una J
recorre del orden de una o dos veces la distancia muñeca-nudillo, y una mano
sostenida solo acumula jitter. **No está medido.** Se calibra en la Fase 2 con una
tanda real: grabar, mirar la distribución de arcos por letra, y ponerlo donde
separe.

También se añade `TOO_MANY_FRAMES`: `dynamic_max_frames` existía en la
configuración pero no se aplicaba, y una grabación dinámica se cierra a mano —
olvidarse es lo más fácil del mundo. El bucle deja de acumular al llegar al tope en
vez de seguir creciendo, para que la muestra siga siendo guardable en lugar de
volverse irrecuperable.

El preview dibuja la barra que corresponde al modo: σ en estáticas, arco en
dinámicas, verde cuando la muestra serviría. Quien graba tiene que saberlo **antes**
de pulsar la tecla; después ya no hay forma de mirar el archivo y averiguarlo.

---

## Decisión 3 — Modo formal y modo prueba

`grabar` distingue dos modos:

- `--sesion-prueba` escribe en `data/raw/pruebas/` y no exige nada. Es para
  encuadrar la cámara y ensayar antes de citar a nadie.
- El modo formal (por defecto) escribe en `data/raw/` y **se niega a arrancar si la
  sección 5 del glosario está vacía**.

Tres personas × 29 letras × dos sesiones contra un glosario que no ha revisado una
persona usuaria de LSM o un intérprete es el error caro de este proyecto. Si una
seña está mal transcrita, las ~60 repeticiones de esa letra son ruido etiquetado y
la única salida es volver a citar a todo el mundo. `ARQUITECTURA.md` §4.11 lo pedía
como norma; esto lo convierte en un cerrojo.

`pruebas/` vive **dentro** de la raíz para compartir consentimiento y calibración
—son del mismo equipo y las mismas personas— pero un nivel más abajo, de modo que
`iter_sample_paths` no lo alcanza. La exclusión es además explícita, porque
depender de que un glob tenga la profundidad justa se rompe la primera vez que
alguien reorganiza una carpeta, y el síntoma sería un modelo entrenado con las
grabaciones de prueba.

### El analizador del glosario se comparte

`lsm.io.glossary` lee la tabla de letras y el registro de validación. Ese código
vivía dentro de `tests/test_vocabulary.py`; se movió al tener un segundo
consumidor, porque dos analizadores del mismo Markdown se desincronizan igual de
callados que las dos fuentes de verdad que pretendían vigilar.

`vocabulary_drift()` informa de **todas** las discrepancias a la vez y no de la
primera: la deriva real de este proyecto afectó a ocho campos en seis letras.

**El test que lo consume no lleva marca `glosario`, y no debe llevarla.** Lo que
hizo que la deriva pasara desapercibida no fue la falta de un test —lo había— sino
que `make test` ya estaba en rojo esperando a una persona, así que un fallo más no
llamó la atención. Tiene que romper `make test-nucleo`, que es el que siempre debe
estar limpio.

---

## Alternativas consideradas

**1. Detectar la lateralidad invertida automáticamente.**
Descartada porque es imposible. No hay ninguna señal en los datos que distinga «la
mano derecha» de «la izquierda espejada»: son el mismo vector. Solo lo sabe quien
está delante de la cámara, y de ahí que la calibración sea un paso humano.

**2. Cubrirlo con golden vectors en vez de con el export.**
No funciona, y es la razón de ser de esta decisión. Los golden vectors reciben la
lateralidad **ya resuelta** como entrada, así que el test de paridad no puede ver
la diferencia. Habría que añadir casos que partieran de una salida cruda del
detector, que es precisamente lo que la especificación no modela.

**3. Caducar la calibración por tiempo.**
Descartada. Añade un umbral arbitrario y una molestia periódica sin tapar ningún
fallo: una calibración se invalida por cambio de cámara o de configuración, y las
dos cosas ya se detectan.

**4. Usar desplazamiento neto de τ en vez de longitud de arco.**
Descartada: rechazaría la `Ñ`, la `Q`, la `Z` y la `X`, que vuelven cerca de donde
empezaron. Es justo el conjunto que el criterio existe para admitir.

**5. Un solo umbral de movimiento con dos lados.**
Tentador —"la ventana debe moverse entre X e Y"— y descartado. σ y el arco no miden
lo mismo: σ es dispersión de la **forma** tras normalizar, ciega al viaje de la
mano; el arco es recorrido de la **muñeca**, ciego al cambio de dedos. Un solo
número no puede sustituir a los dos.

**6. Avisar en vez de bloquear.**
Descartada para los tres cerrojos, por el mismo motivo: los tres fallos que tapan
no dejan rastro en el dataset, así que el aviso sería el único momento en que
alguien podría enterarse — y es el momento en que menos atención hay.

**7. Escribir las sesiones de prueba fuera de `data/raw/`.**
Descartada. Tendrían que duplicar el registro de consentimiento y el de
calibración, que son del mismo equipo y las mismas personas. Un nivel más abajo
resuelve el aislamiento sin duplicar nada.

---

## Consecuencias

**A favor**

- La convención de lateralidad deja de ser una convención tácita: viaja en cada
  muestra, viaja en cada modelo, y se rechaza al cargar si no coincide.
- La Fase 7 tiene un mecanismo que fuerza el acuerdo con MediaPipe JS. Antes no
  había ninguno.
- Las ocho letras dinámicas dejan de grabarse sin filtro.
- Los tres errores irreversibles de la captura son ahora rechazos comprobables, y
  los tres tienen test.

**En contra**

- Grabar exige un paso previo por cámara (`calibrar`). Es un minuto, una vez.
- `min_trajectory_arc` es un valor **razonado y no medido**, y rechazará muestras
  legítimas si queda alto o dejará pasar trazos flojos si queda bajo. Se calibra en
  la Fase 2; hasta entonces, quien grabe debe mirar la barra y avisar si molesta.
- `SAMPLE_SCHEMA_VERSION` pasa a 2 y `CAPTURE_SPEC_VERSION` a 2. Las muestras del
  esquema 1 se rechazan al cargarse. No hay ninguna todavía, que es justo por qué
  este era el momento.

**Lo que sigue bloqueando la primera sesión formal**

1. **La sección 5 del glosario está vacía** (`PENDIENTE-HUMANO G`). Falta la
   revisión con una persona usuaria de LSM o un intérprete. Es humano y ahora
   además es un cerrojo.
2. **Ninguna cámara está calibrada.** Es un minuto con `lsm-capture calibrar`, pero
   hay que hacerlo delante de la cámara que se vaya a usar.

**Cómo se cambia**

`HANDEDNESS_CONVENTION` no se cambia sin invalidar todos los modelos exportados y
re-canonizar el dataset. Lo segundo es barato —los landmarks crudos no dependen de
la convención, solo su etiqueta— pero hay que hacerlo a propósito y a la vez que lo
primero.

`min_trajectory_arc` sí se ajusta libremente: es un umbral de configuración, no
entra en ningún vector, y cambiarlo no invalida nada ya grabado. Al ajustarlo,
incrementar `CAPTURE_SPEC_VERSION` y anotar con qué datos se calibró.
