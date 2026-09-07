# Glosario LSM — Alfabeto dactilológico

**Estado: PLANTILLA. Debe completarse antes de iniciar la Fase 1.**

Este documento es la referencia normativa del proyecto. Define qué señas se
reconocen, cuáles implican movimiento y cómo se ejecuta cada una correctamente.

\---

## 1\. Fuente de referencia

**Fuente primaria**

> Serafín de Fleischmann, M. E. y González Pérez, R. \*Manos con voz. Diccionario de
> Lengua de Señas Mexicana\*. CONAPRED / Libre Acceso A.C. Abecedario: páginas 15-19.

* PDF alojado por la Dirección de Educación Especial (SEP):
`https://educacionespecial.sep.gob.mx/storage/recursos/2023/05/xzrfl019nV-4Diccionario\_lengua\_%20Senas.pdf`
* Ficha institucional: `https://www.conapred.org.mx/publicaciones/manos-con-voz-diccionario-de-lengua-de-senas-mexicana/`

**Fuentes secundarias**

* *DIELSEME*, Dirección de Educación Especial, SEP — contexto lingüístico y léxico.
* Revisión crítica en SciELO (`S1010-29142014000300005`) — limitaciones lexicográficas
de ambas obras. Citar en el marco teórico.

**Convención de la fuente:** las flechas en las fotografías indican dirección del
movimiento; las flechas de doble punta indican movimiento de ida y vuelta. Toda letra
con flecha en el diccionario se marca `es\_dinamica: true` en este glosario.

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
`Ñ` → `ENIE`, `LL` → `DOBLE\_L`, `RR` → `DOBLE\_R`.
* `es\_dinamica` — `true` si la ejecución requiere movimiento.
* `trayectoria` — solo si es dinámica: descripción del recorrido y si es de ida y
vuelta.
* `confundible\_con` — letras de configuración similar. Alimenta el análisis de la
matriz de confusión de la Fase 2.
* `pagina` — página de la fuente donde se verificó.

|label|letra|es\_dinamica|descripción de la configuración|trayectoria|confundible\_con|página|
|-|-|-|-|-|-|-|
|A|A|false||—|E|15|
|B|B|false||—|F|15|
|C|C|false||—|O|15|
|D|D|false||—|U,R|15|
|E|E|false||—|A|15|
|F|F|false||—|B|15|
|G|G|false||—|H|16|
|H|H|false||—|G|16|
|I|I|false||—|J|16|
|J|J|true||hacia abajo y vuelta en u|I|16|
|K|K|true||izquierda a derecha o derecha a izquierda|P|16|
|L|L|false||—|E|16|
|DOBLE\_L|LL|false||—|E|16|
|M|M|false||—|N|17|
|N|N|false||—|M|17|
|ENIE|Ñ|true||rotacion de ida y vuelta|N|17|
|O|O|false||—|C|17|
|P|P|false||—|K|17|
|Q|Q|true||rotacion de ida y vuelta||17|
|R|R|false||—|U|18|
|DOBLE\_R|RR|false||—||18|
|S|S|false||—|T|18|
|T|T|false||—|S|18|
|U|U|false||—|R,D|18|
|V|V|false||—|W|18|
|W|W|false||—|V|18|
|X|X|true||de un lado al otro horizontalmente, de ida y vuelta||19|
|Y|Y|false||—||19|
|Z|Z|true||se dibuja la letra z||19|

**Al llenar, prestar atención especial a:**

* Las letras de configuración de puño cerrado con variación de pulgar. Son las que
concentrarán los errores del clasificador. Anotarlas todas en `confundible\_con`
desde ahora, antes de tener datos, y contrastar después contra la matriz de
confusión real.
* Marcar `es\_dinamica` según las flechas del diccionario, no según intuición.
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
  "schema\_version": 1,
  "reference\_source": "Manos con voz, CONAPRED/Libre Acceso, pp. 15-19",
  "signs": \[
    {
      "label": "A",
      "display": "A",
      "es\_dinamica": false,
      "asset": "a.webp",
      "asset\_type": "image",
      "descripcion": "Redacción propia de la configuración manual.",
      "verificado\_contra": "Manos con voz, p. 15",
      "capturado\_en": "session\_2026\_03\_10\_signer\_01",
      "revisado\_por": "nombre del intérprete o persona usuaria"
    },
    {
      "label": "J",
      "display": "J",
      "es\_dinamica": true,
      "asset": "j.webp",
      "asset\_type": "animation",
      "duracion\_ms": 1200,
      "descripcion": "Redacción propia, incluyendo el recorrido.",
      "verificado\_contra": "Manos con voz, p. 16",
      "capturado\_en": "session\_2026\_03\_10\_signer\_01",
      "revisado\_por": "nombre del intérprete o persona usuaria"
    }
  ]
}
```

**Reglas de validación del manifest** (test automatizado en la Fase 4):

1. Existe exactamente una entrada por `label` de la tabla de la sección 3.
2. Todo archivo referenciado en `asset` existe en disco.
3. Toda entrada con `es\_dinamica: true` tiene `asset\_type: "animation"` y
`duracion\_ms`. Una imagen fija no puede representar una letra con movimiento.
4. Ninguna entrada tiene `verificado\_contra` o `revisado\_por` vacíos.

