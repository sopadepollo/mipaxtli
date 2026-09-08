# Glosario LSM — Alfabeto dactilológico

**Estado: PLANTILLA. Debe completarse antes de iniciar la Fase 1.**

Este documento es la referencia normativa del proyecto. Define qué señas se
reconocen, cuáles implican movimiento y cómo se ejecuta cada una correctamente.

\---

## 1\. Fuente de referencia

**Fuente primaria**

> Serafín de Fleischmann, M. E. y González Pérez, R. \\\\\\\*Manos con voz. Diccionario de
> Lengua de Señas Mexicana\\\\\\\*. CONAPRED / Libre Acceso A.C. Abecedario: páginas 15-19.

* PDF alojado por la Dirección de Educación Especial (SEP):
`https://educacionespecial.sep.gob.mx/storage/recursos/2023/05/xzrfl019nV-4Diccionario\\\\\\\_lengua\\\\\\\_%20Senas.pdf`
* Ficha institucional: `https://www.conapred.org.mx/publicaciones/manos-con-voz-diccionario-de-lengua-de-senas-mexicana/`

**Fuentes secundarias**

* *DIELSEME*, Dirección de Educación Especial, SEP — contexto lingüístico y léxico.
* Revisión crítica en SciELO (`S1010-29142014000300005`) — limitaciones lexicográficas
de ambas obras. Citar en el marco teórico.

**Convención de la fuente:** las flechas en las fotografías indican dirección del
movimiento; las flechas de doble punta indican movimiento de ida y vuelta. Toda letra
con flecha en el diccionario se marca `es\\\\\\\_dinamica: true` en este glosario.

\---

## 2\. Advertencias que condicionan el proyecto

**LSM no es ASL.** La mayor parte del material que circula en internet etiquetado
como "abecedario en lengua de señas" es ASL (estadounidense). Son alfabetos
distintos. Todo asset y toda seña debe verificarse contra la fuente primaria. Un
proyecto que presenta ASL como LSM queda descalificado ante cualquier persona
usuaria.

**Variación regional.** La LSM presenta variantes dialectales entre regiones de
México. La propia SEP reconoce que la recopilación de variantes quedó pendiente en
DIELSEME. Decisión del proyecto: se documenta la variante de la fuente primaria y se
registra en los metadatos de cada muestra la región de quien firma.
Ver `docs/adr/000X-variante-dialectal.md`.

**Validación humana obligatoria.** Este glosario debe revisarse con una persona
usuaria de LSM o un intérprete certificado antes de cerrar la Fase 1. Registrar
nombre, fecha y observaciones en la sección 5.

\---

## 3\. Tabla de letras

Llenar transcribiendo de las páginas 15-19 de la fuente primaria. La descripción se
redacta **con palabras propias** a partir de la descripción del diccionario, no se
copia literalmente.

Columnas:

* `label` — identificador usado en código y dataset. Mayúsculas, sin acentos.
`Ñ` → `ENIE`, `LL` → `DOBLE\\\\\\\_L`, `RR` → `DOBLE\\\\\\\_R`.
* `es\\\\\\\_dinamica` — `true` si la ejecución requiere movimiento.
* `trayectoria` — solo si es dinámica: descripción del recorrido y si es de ida y
vuelta.
* `confundible\\\\\\\_con` — letras de configuración similar. Alimenta el análisis de la
matriz de confusión de la Fase 2.
* `pagina` — página de la fuente donde se verificó.

|label|letra|es\_dinamica|descripción de la configuración|trayectoria|confundible\_con|página|
|-|-|-|-|-|-|-|
|A|A|false|Mano cerrada, se muestran las uñas<br />y se estira el dedo pulgar hacia un lado. La<br />palma mira al frente|—|E,L,DOBLE\_L|15|
|B|B|false|Dedos índice, medio, anular y meñique se<br />estiran unidos y el pulgar se dobla dirección a <br />la palma, la cual mira al frente|—|F|15|
|C|C|false|Dedos índice, medio, anular y meñique se<br />mantienen unidos y en posición cóncava;<br />el pulgar también se pone de esa forma. La<br />palma mira a un lado|—|O|15|
|D|D|false|Dedos medio, anular, meñique y pulgar se<br />unen por las puntas y el dedo índice se estira.<br />La palma mira al frente|—|U,R,DOBLE\_R|15|
|E|E|false|Dedos completamente doblados, se<br />muestran las uñas. La palma mira al frente|—|A,L,DOBLE\_L|15|
|F|F|false|Mano abierta y los dedos unidos,<br />se dobla el índice hasta que su parte lateral<br />toque la yema del pulgar. La palma mira a un<br />lado|—|B|15|
|G|G|false|Mano cerrada y los dedos índice y pulgar estirados. La palma mira  adentro|—|H|16|
|H|H|false|Mano cerrada y los dedos índice y<br />medio estirados y unidos, se extiende el<br />dedo pulgar señalando hacia arriba. La palma<br />mira adentro|—|G|16|
|I|I|false|Mano cerrada, el dedo meñique se<br />estira señalando hacia arriba. La palma se<br />pone de lado|—|J|16|
|J|J|true|Mano cerrada, el dedo meñique bien<br />estirado señalando hacia arriba y la palma a<br />un lado dibuja una j en el aire|dibuja una j en el aire|I|16|
|K|K|true|Se cierra la mano con los dedos índice, medio<br />y pulgar estirados. La yema del pulgar se<br />pone entre el índice y el medio. Se mueve la<br />muñeca hacia arriba|se mueve la muneca hacia arriba|P|16|
|L|L|false|Mano cerrada y los dedos índice y<br />pulgar estirados, se forma una l. La palma<br />mira al frente|—|E,A,DOBLE\_L|16|
|DOBLE\_L|LL|true|Mano cerrada y los dedos índice y<br />pulgar estirados, se forma una l. La palma<br />mira al frente|movimientos de adelante a atras u horizontales|E,A,L|16|
|M|M|false|Mano cerrada, se ponen los dedos<br />índice, medio y anular sobre el pulgar|—|N|17|
|N|N|false|Mano cerrada, se ponen los dedos<br />índice y medio sobre el pulgar|—|M,ENIE|17|
|ENIE|Ñ|true|Mano cerrada, se ponen los dedos<br />índice y medio sobre el pulgar. Se mueve la<br />muñeca a los lados|rotacion de ida y vuelta|N,Q|17|
|O|O|false|Con la mano se forma una letra o. Todos los<br />dedos se tocan por las puntas|—|C|17|
|P|P|false|Mano cerrada y los dedos índice, medio<br />y pulgar estirados, se pone la yema del pulgar<br />entre el índice y el medio|—|K|17|
|Q|Q|true|Mano cerrada, se ponen los dedos<br />índice y pulgar en posición de garra. La palma<br />mira hacia abajo, y se mueve la muñeca hacia<br />los lados|rotacion de ida y vuelta|ENIE|17|
|R|R|false|Mano cerrada, se estiran y entrelazan<br />los dedos índice y medio. La palma mira al<br />frente|—|U,D,DOBLE\_R|18|
|DOBLE\_R|RR|true|Mano cerrada, se estiran y entrelazan<br />los dedos índice y medio. La palma mira al<br />frente|movimientos de adelante a atras u horizontales|U,D,R|18|
|S|S|false|Mano cerrada, se pone el pulgar sobre<br />los otros dedos. La palma mira al frente|—|T|18|
|T|T|false|Mano cerrada, el pulgar se pone entre<br />el índice y el medio. La palma mira al frente|—|S|18|
|U|U|false|Mano cerrada, se estiran los dedos<br />índice y medio unidos. La palma mira al frente|—|R,DOBLE\_R,D|18|
|V|V|false|Mano cerrada, se estiran los dedos<br />índice y medio separados. La palma mira al<br />frente|—|W|18|
|W|W|false|Mano cerrada, se estiran los dedos<br />índice, medio y anular separados. La palma<br />mira al frente|—|V|18|
|X|X|true|Mano cerrada, el índice y el pulgar<br />en posición de garra y la palma dirigida a un<br />lado, se realiza un movimiento al frente y de<br />regreso|movimiento al frente y de<br />regreso||19|
|Y|Y|false|Mano cerrada, se estira el meñique<br />y el pulgar. La palma mira hacia dentro|—||19|
|Z|Z|true|Mano cerrada, el dedo índice estirado<br />y la palma al frente, se dibuja una letra z en<br />el aire|dibuja una letra z en<br />el aire||19|

### PENDIENTE-HUMANO — estado de la revisión

Revisión del 2026-09-07, tras el llenado de la tabla. La verificación automática
vive en `tests/test_vocabulary.py`; lo que sigue son las cuestiones que ninguna
prueba puede decidir por su cuenta.

#### Resueltas en el llenado

- **Simetría de `confundible_con`.** Los cuatro pares asimétricos (D↔R, E↔L,
  E↔LL, N↔Ñ) están cerrados. La comprobación automática pasa.
- **Las 29 descripciones de configuración manual.** Completas.
- **`LL` y `RR` eran sospechosas como estáticas.** Confirmado: ahora son dinámicas
  y listan a `L` y `R` como confundibles. Coinciden con `ARQUITECTURA.md` §1, que
  ya daba `RR` por dinámica.
- **Dirección ambigua de la `K`.** Era "izquierda a derecha o derecha a izquierda";
  ahora es un movimiento único de muñeca hacia arriba. Con una sola dirección
  canónica, el DTW ve una sola clase.
- **`Ñ` y `Q` compartían trayectoria sin listarse.** Ya se listan mutuamente.

#### Abiertas

**PENDIENTE-HUMANO A — `LL` y `RR` repiten la ambigüedad que la `K` acaba de
perder.** Su trayectoria es "movimientos de adelante a atrás **u** horizontales":
dos recorridos distintos para una misma etiqueta. Es el mismo problema que tenía la
`K`, y tiene el mismo efecto: dos firmas de τ diferentes para la misma clase, que
el DTW tratará como dos señas. Hay que elegir una canónica, o aceptar las dos y
guardar plantillas separadas en la Fase 5. Verificar pp. 16 y 18.

**PENDIENTE-HUMANO B — cuatro letras dinámicas tienen la misma configuración de
mano que una estática.** No es un error del glosario: es lo que dice la fuente, y
conviene tenerlo escrito porque condiciona el diseño.

| Par | Configuración | Qué las separa |
|---|---|---|
| `L` / `LL` | idéntica, palabra por palabra | solo el movimiento |
| `R` / `RR` | idéntica, palabra por palabra | solo el movimiento |
| `N` / `Ñ` | idéntica | solo el movimiento de muñeca |
| `I` / `J` | meñique estirado en ambas | solo el trazo |

Consecuencia directa: para estas cuatro el vector de forma de 42 componentes es
prácticamente el mismo, y **el canal de trayectoria del §3 es lo único que las
distingue**. Dos cosas se siguen de ahí:

1. `config.features.trajectory_weight` no es un ajuste fino, es lo que hace
   reconocibles a cuatro letras. Calibrarlo con datos reales es obligatorio.
2. El enrutamiento estático/dinámico **no puede ser "corre ambos y toma la mayor
   confianza"**: el clasificador estático dirá `L` con confianza alta ante una
   `LL`, porque para él son la misma mano. Ver
   `docs/adr/0004-contrato-de-segmentacion.md`.

**PENDIENTE-HUMANO C — `X` y `Q` comparten la configuración de garra y no se
listan.** Ambas describen índice y pulgar "en posición de garra"; se diferencian
por la orientación de la palma (`X` de lado, `Q` hacia abajo) y por el movimiento.
El problema es que la tubería rota la mano para alinear la palma con el eje +Y y
descarta la profundidad, así que **parte de esa diferencia de orientación se pierde
en la normalización**. Revisar si deben listarse como confundibles. Verificar
pp. 17 y 19.

**PENDIENTE-HUMANO D — `I` y `Y` se diferencian solo por el pulgar y no se
listan.** `I` estira el meñique; `Y` estira el meñique y el pulgar. Es exactamente
el caso que `ARQUITECTURA.md` §4.8 anticipa como fuente principal de error.
Verificar pp. 16 y 19.

**PENDIENTE-HUMANO E — `CH` ausente sin justificación.** La tabla incluye `LL` y
`RR` pero no `CH`, que fue dígrafo del español hasta 1994. Si la fuente primaria no
lo trae, anotarlo explícitamente; si lo trae, falta la fila. Verificar pp. 15-19.

**PENDIENTE-HUMANO F — `Z` traza un recorrido mucho más largo que el resto.**
Riesgo conocido: con `config.dtw.band_radius = 6` sobre 24 frames, la banda de
Sakoe-Chiba puede quedar corta para una seña que ocupa toda la ventana, y la `Z`
saldría peor que las demás sin motivo aparente. **No cambiar el valor ahora** —se
calibra con datos reales en la Fase 2—, solo mirar la `Z` aparte al leer la primera
matriz de confusión.

**PENDIENTE-HUMANO G — falta la revisión con una persona usuaria de LSM.** La
sección 5 sigue vacía. El glosario está transcrito de la fuente primaria, que es
condición necesaria pero no suficiente: `ARQUITECTURA.md` §4.11 pide validación
humana antes de presentar el proyecto.

\---

**Al llenar, prestar atención especial a:**

* Las letras de configuración de puño cerrado con variación de pulgar. Son las que
concentrarán los errores del clasificador. Anotarlas todas en `confundible\\\\\\\_con`
desde ahora, antes de tener datos, y contrastar después contra la matriz de
confusión real.
* Marcar `es\\\\\\\_dinamica` según las flechas del diccionario, no según intuición.
* Los dígrafos LL y RR: decidir explícitamente si entran al alcance y registrarlo en
un ADR. Si entran, definir cómo se resuelven en el buffer de deletreo (¿"RR" es un
símbolo o dos "R" consecutivas?).

\---

## 4\. Clase negativa

Además de las letras, el dataset incluye una clase `NONE` con:

* Mano relajada y en reposo.
* Transiciones entre letras.
* Gestos cotidianos no pertenecientes al alfabeto (saludar, señalar, mano abierta).

Sin esta clase, el clasificador asigna siempre una de las 27 letras aunque la persona
no esté firmando. Ver `ARQUITECTURA.md` §4.4.

\---

## 5\. Registro de validación

|Fecha|Revisor|Rol|Letras revisadas|Observaciones|
|-|-|-|-|-|
||||||

\---

## Apéndice — Esquema de `assets/signs/manifest.json`

Los assets se generan a partir de las propias sesiones de captura de la Fase 1, no se
descargan. Esto evita problemas de licencia, garantiza consistencia visual con lo que
el sistema espera ver, y no cuesta trabajo adicional: son las mismas grabaciones.

```json
{
  "schema\\\\\\\_version": 1,
  "reference\\\\\\\_source": "Manos con voz, CONAPRED/Libre Acceso, pp. 15-19",
  "signs": \\\\\\\[
    {
      "label": "A",
      "display": "A",
      "es\\\\\\\_dinamica": false,
      "asset": "a.webp",
      "asset\\\\\\\_type": "image",
      "descripcion": "Redacción propia de la configuración manual.",
      "verificado\\\\\\\_contra": "Manos con voz, p. 15",
      "capturado\\\\\\\_en": "session\\\\\\\_2026\\\\\\\_03\\\\\\\_10\\\\\\\_signer\\\\\\\_01",
      "revisado\\\\\\\_por": "nombre del intérprete o persona usuaria"
    },
    {
      "label": "J",
      "display": "J",
      "es\\\\\\\_dinamica": true,
      "asset": "j.webp",
      "asset\\\\\\\_type": "animation",
      "duracion\\\\\\\_ms": 1200,
      "descripcion": "Redacción propia, incluyendo el recorrido.",
      "verificado\\\\\\\_contra": "Manos con voz, p. 16",
      "capturado\\\\\\\_en": "session\\\\\\\_2026\\\\\\\_03\\\\\\\_10\\\\\\\_signer\\\\\\\_01",
      "revisado\\\\\\\_por": "nombre del intérprete o persona usuaria"
    }
  ]
}
```

**Reglas de validación del manifest** (test automatizado en la Fase 4):

1. Existe exactamente una entrada por `label` de la tabla de la sección 3.
2. Todo archivo referenciado en `asset` existe en disco.
3. Toda entrada con `es\\\\\\\_dinamica: true` tiene `asset\\\\\\\_type: "animation"` y
`duracion\\\\\\\_ms`. Una imagen fija no puede representar una letra con movimiento.
4. Ninguna entrada tiene `verificado\\\\\\\_contra` o `revisado\\\\\\\_por` vacíos.

