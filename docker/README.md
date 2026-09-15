# Docker

Dos cosas conviven en este proyecto y tienen necesidades opuestas:

- **Lo que no toca hardware** — tests, entrenamiento, evaluación, generación de
  golden vectors. Se beneficia del contenedor: entorno reproducible, mismo
  resultado en tu máquina y en CI.
- **Lo que sí toca hardware** — la captura del dataset y la demo en vivo.
  Necesitan la webcam, y ahí Docker deja de ayudar.

La arquitectura está pensada para que esa frontera sea limpia: `src/lsm/io/camera.py`
es el único módulo que habla con la cámara y `src/lsm/io/hands.py` el único que
puede importar MediaPipe. Todo lo demás son funciones puras sobre arreglos. Por
eso lo primero se puede contenerizar sin pelear con nada.

## Uso

```bash
# Criterio de aceptación de la Fase 0: ruff + mypy strict + pytest.
docker compose -f docker/docker-compose.yml run --rm test

# Iterar montando el código, sin reconstruir la imagen.
docker compose -f docker/docker-compose.yml run --rm dev

# Regenerar los golden vectors dentro del contenedor.
docker compose -f docker/docker-compose.yml run --rm golden
```

Ninguno de esos comandos necesita cámara, MediaPipe ni dataset.

---

## (a) Cámara en Linux: passthrough de `/dev/video0`

En Linux la webcam es un archivo de dispositivo y el contenedor puede recibirlo
directamente. Esto aplica a las fases posteriores (captura y demo), **no** a la
Fase 0.

```bash
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/data:/app/data" \
  lsm-translator make capture
```

O como servicio de compose, para cuando exista `capture.py`:

```yaml
  capture:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    devices:
      - /dev/video0:/dev/video0
    environment:
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ../data:/app/data
    command: ["capture"]
```

Detalles que suelen morder:

- **Qué dispositivo es la cámara.** No siempre es `video0`. `v4l2-ctl --list-devices`
  lo dice; muchas webcams exponen dos nodos y solo uno entrega video.
- **Permisos.** El dispositivo pertenece al grupo `video`. Si el usuario del
  contenedor no está en ese grupo, `open()` falla con un error poco claro.
  Se resuelve con `--group-add video` o corriendo como root, con lo que eso
  implica.
- **La ventana de preview.** El CLI de captura muestra los landmarks dibujados, y
  para eso el contenedor necesita acceso al servidor X (`DISPLAY` y el socket
  montado) o Wayland con `xhost`/XWayland. Es la parte que más tiempo consume y
  la que menos valor aporta.
- **La imagen que se le pasa a MediaPipe va sin espejar.** Es una regla del
  contrato (`docs/feature-spec.md` §0.3), no una cuestión de Docker, pero se
  rompe con facilidad al montar el preview.

## (b) Windows y macOS: la captura se ejecuta fuera del contenedor

**En Windows y macOS no hay passthrough de webcam, y no vale la pena pelearlo.**

La razón es estructural, no un ajuste que falte encontrar: en esos sistemas Docker
no corre sobre el kernel del host. Docker Desktop levanta una máquina virtual
Linux —WSL2 en Windows, un hipervisor en macOS— y los contenedores viven dentro de
ella. La webcam es un dispositivo del host, y la VM no la ve:

- **Windows.** WSL2 no expone dispositivos USB al kernel Linux por omisión: sin
  más, no hay `/dev/video*`. Se puede intentar con `usbipd-win`, pero el camino
  sigue siendo frágil por otros motivos: hay que enlazar el dispositivo en cada
  arranque, la latencia sube, y **mientras la cámara está conectada a WSL deja de
  funcionar en Windows** — se la lleva entera, no la comparte.

  Un matiz que conviene tener actualizado: los kernels recientes de WSL2 **sí
  traen los módulos** de video compilados. Comprobado sobre
  `6.6.87.2-microsoft-standard-WSL2`:

  ```
  /lib/modules/$(uname -r)/kernel/drivers/media/usb/uvc/uvcvideo.ko
  /lib/modules/$(uname -r)/kernel/drivers/media/v4l2-core/videodev.ko
  ```

  Es decir, el obstáculo ya no es que el kernel no sepa hablar con una webcam,
  sino que el dispositivo no llega hasta él. Sigue sin valer la pena: el camino de
  abajo funciona sin instalar nada con privilegios de administrador y sin quitarle
  la cámara al resto del sistema.
- **macOS.** El acceso a la cámara está mediado por el framework de privacidad de
  macOS, que concede permiso a **aplicaciones firmadas del host**. No existe una
  ruta soportada para que un proceso dentro de la VM de Docker Desktop obtenga
  ese permiso.

### Cómo se trabaja entonces

Se parte el flujo por la misma frontera que ya define la arquitectura:

| Tarea | Dónde corre | Por qué |
|---|---|---|
| Tests, lint, tipos | Contenedor o host | No tocan hardware |
| Entrenamiento y evaluación | Contenedor | Reproducible; solo lee `data/` |
| Golden vectors | Contenedor o host | Código puro y determinista |
| **Calibración de la cámara** | **Host, fuera de Docker** | Necesita webcam y un ojo humano |
| **Captura del dataset** | **Host, fuera de Docker** | Necesita webcam |
| **Demo en vivo** | **Host, fuera de Docker** | Necesita webcam |

En el host, sin contenedor:

```bash
uv sync --extra capture           # MediaPipe y OpenCV, que no van en la imagen
uv run lsm-capture calibrar       # una vez por cámara
uv run lsm-capture grabar ...
```

Las muestras quedan en `data/raw/`, que es un directorio del proyecto. El
contenedor de entrenamiento lo monta como volumen y ya. El dataset son landmarks
en JSON, no video: se mueve entre máquinas sin problema.

### El caso concreto: repositorio en WSL, captura desde Windows

Es la combinación más común al desarrollar en Windows, y funciona sin mover el
repositorio ni duplicarlo. **Receta comprobada** (MediaPipe 1.0.1, OpenCV 5.0,
Python 3.13, webcam integrada por el backend `MSMF`):

```powershell
# 1. Un entorno de Windows, fuera del repositorio. No lo referencia nada del
#    proyecto: es solo el intérprete que tiene acceso a la cámara.
py -m venv $HOME\lsm-win
& $HOME\lsm-win\Scripts\python.exe -m pip install mediapipe opencv-python pydantic pyyaml

# 2. El repositorio se lee por UNC. `python.exe` acepta una ruta UNC como
#    directorio de trabajo cuando se lanza desde PowerShell, así que las rutas
#    relativas de config.yaml resuelven solas.
$repo = "\\wsl.localhost\Ubuntu\home\<usuario>\mipaxtli"
Set-Location $repo
$env:PYTHONPATH = "$repo\src"

# 3. El modelo del detector (~8 MB) se descarga una vez, desde WSL:
#       make model
#    Queda en data/models/ y Windows lo lee por la misma ruta UNC.

# 4. Calibrar (una vez por cámara) y grabar.
& $HOME\lsm-win\Scripts\python.exe -m lsm.cli.capture calibrar --confirmado-por "tu nombre"
& $HOME\lsm-win\Scripts\python.exe -m lsm.cli.capture grabar --sesion-prueba --firmante s01 ...
```

Se invoca con `python -m lsm.cli.capture` y no con `lsm-capture` porque el
proyecto no está instalado en ese entorno: solo está en el `PYTHONPATH`. Es
deliberado — así el entorno de Windows no tiene que mantenerse sincronizado con
`pyproject.toml`, y `uv` sigue siendo la única fuente de verdad de las
dependencias dentro de WSL.

**Detalle que muerde:** la clave del registro de calibración incluye el backend
de OpenCV (`MSMF:0@1280x720` en Windows, `V4L2:0@1280x720` en Linux). Calibrar
desde Windows y grabar desde WSL son, para `lsm.io.calibration`, dos cámaras
distintas, y la segunda pedirá su propia calibración. **Es correcto y a
propósito**: el mismo dispositivo por dos backends no entrega lo mismo, y la
comprobación de lateralidad hecha con uno no dice nada del otro.

Esto no es una limitación del diseño, es el diseño: `ARQUITECTURA.md` §4.9 lo
decide explícitamente para no gastar el tiempo del proyecto peleando con Docker
Desktop por la webcam.

### Avisos de MediaPipe que no son problemas tuyos

MediaPipe escribe en `stderr` desde su capa de C++, y algunos mensajes suenan a
error sin serlo. El que aparece en cada sesión de captura:

```
Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI.
Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
```

Sale de `landmark_projection_calculator.cc`, **dentro del grafo que MediaPipe
distribuye ya compilado en el `.task`**. No se puede corregir desde la API de
Python: no hay ningún parámetro de `HandLandmarkerOptions` que lo controle. Se
reporta desde la versión 0.10.15 en los tres landmarkers (manos, cara, pose) y el
modelo funciona igual — ver [google-ai-edge/mediapipe#6040] y [#5639].

Aparece solo cuando **hay una mano en el encuadre**, porque el cálculo que lo emite
es el que proyecta los landmarks del recorte al cuadro completo. Dicho de otro
modo: si lo ves, es que la detección está funcionando.

Cómo comprobar que efectivamente es inofensivo, sin fiarse de esto: en el preview,
mirar que el esqueleto verde se dibuje **sobre** la mano y la siga al moverla. Si
los puntos caen donde deben, la proyección es correcta y el mensaje es ruido.

[google-ai-edge/mediapipe#6040]: https://github.com/google-ai-edge/mediapipe/issues/6040
[#5639]: https://github.com/google-ai-edge/mediapipe/issues/5639

## CI

CI corre el servicio `test`, que no necesita cámara ni dataset. Ese es justamente
el criterio de aceptación de la Fase 0: la suite completa pasa en una máquina sin
hardware de video y sin MediaPipe instalado.
