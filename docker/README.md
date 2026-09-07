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

- **Windows.** WSL2 no expone dispositivos USB al kernel Linux por omisión. Se
  puede intentar con `usbipd-win`, pero el camino es frágil: hay que enlazar el
  dispositivo en cada arranque, la latencia sube, y el driver `v4l2` de la VM
  tiene que aceptar la cámara, cosa que muchas webcams integradas no hacen. Las
  cámaras integradas de laptop suelen no funcionar ni con eso.
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
| **Captura del dataset** | **Host, fuera de Docker** | Necesita webcam |
| **Demo en vivo** | **Host, fuera de Docker** | Necesita webcam |

En el host, sin contenedor:

```bash
uv sync
uv run lsm-capture     # cuando exista (Fase 1)
```

Las muestras quedan en `data/raw/`, que es un directorio del proyecto. El
contenedor de entrenamiento lo monta como volumen y ya. El dataset son landmarks
en JSON, no video: se mueve entre máquinas sin problema.

Esto no es una limitación del diseño, es el diseño: `ARQUITECTURA.md` §4.9 lo
decide explícitamente para no gastar el tiempo del proyecto peleando con Docker
Desktop por la webcam.

## CI

CI corre el servicio `test`, que no necesita cámara ni dataset. Ese es justamente
el criterio de aceptación de la Fase 0: la suite completa pasa en una máquina sin
hardware de video y sin MediaPipe instalado.
