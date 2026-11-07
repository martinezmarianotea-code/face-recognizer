# Face Recognizer

Aplicación experimental para registrar rostros y probar reconocimiento facial en una computadora local. Usa **MediaPipe** para detectar caras y **LBPH de OpenCV** para comparar los recortes registrados.

> No es un sistema de autenticación ni de seguridad. Su precisión, rendimiento y sesgo no se han evaluado de forma sistemática.

## Funciones

- Interfaz web local adaptable a escritorio y móvil.
- Captura de recortes desde la cámara del navegador.
- Entrenamiento y reconocimiento ejecutados localmente con Python.
- Análisis de imágenes y cámara en vivo.
- Administración de perfiles, recortes y modelo desde el navegador.
- Interfaces adicionales de escritorio con Tkinter y de línea de comandos.

## Requisitos

- Python 3.10–3.12.
- Navegador actualizado; para capturar video, concede permiso de cámara a `localhost` o `127.0.0.1`.
- Cámara web para capturar rostros o reconocer en vivo.
- Conexión a internet durante la instalación inicial para descargar los modelos de MediaPipe.

## Instalación

```bash
git clone https://github.com/martinezmarianotea-code/face-recognizer.git
cd face-recognizer
python -m venv .venv
```

Activa el entorno virtual:

```bash
# Linux o macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instala las dependencias y descarga los modelos necesarios:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python 1_setup.py
```

## Interfaz web local

```bash
python web_app.py
```

Abre `http://127.0.0.1:8765` en el navegador. Para cerrar el servidor, vuelve a la terminal y pulsa `Ctrl+C`.

```bash
python web_app.py --no-browser
python web_app.py --port 9000
```

El servidor escucha solo en el equipo local. No lo expongas a una red ni lo publiques detrás de un proxy. El navegador pide permiso antes de activar la cámara. Al registrar, se guarda el recorte facial; al reconocer una imagen o usar la cámara en vivo, los fotogramas se procesan en memoria.

## Interfaz de escritorio

```bash
python app.py
```

Requiere un entorno gráfico compatible con Tkinter y OpenCV.

## Línea de comandos

### Registrar recortes

```bash
python 2_collect_faces.py --name "Persona" --source webcam --max 30
python 2_collect_faces.py --name "Persona" --source ./mis_fotos --max 30
python 2_collect_faces.py --name "Persona" --source ./video.mp4 --max 50
```

En la captura por webcam, `Espacio` guarda una imagen, `A` alterna la captura automática y `Q` termina. Se recomienda registrar al menos 15 imágenes variadas y nítidas por persona.

Los recortes se guardan en `data/known_faces/`. El límite es de 500 imágenes por persona.

### Entrenar el reconocedor

```bash
python 3_train.py
```

### Reconocer

```bash
python 4_recognize.py
python 4_recognize.py --source ./foto.jpg
python 4_recognize.py --source ./video.mp4 --save
python 4_recognize.py --source ./imagenes/ --save
```

`--no-mesh` desactiva la malla facial, `--no-pose` desactiva la estimación de pose y `--save` guarda resultados en `output/`. También está disponible el menú de terminal con `python menu.py`.

## Privacidad y uso responsable

Las imágenes, los modelos entrenados y los resultados pueden revelar información biométrica. El proyecto los mantiene localmente y los excluye del control de versiones. Los archivos locales no están cifrados; protege tu cuenta del sistema y elimina `data/`, `models/` y `output/` cuando ya no los necesites.

Solicita consentimiento antes de capturar o procesar rostros. No uses el resultado para decisiones críticas, control de acceso o identificación a gran escala. La distancia LBPH es una medida heurística, no una probabilidad de identidad.

## Limitaciones

- LBPH depende de la pose, la iluminación, el encuadre y la calidad de las imágenes.
- El umbral de reconocimiento puede requerir ajuste para cada entorno.
- El rendimiento depende del procesador y del tamaño de imagen.
- La cámara web está disponible solo desde el origen local.

## Estructura

```text
web/                  Interfaz web
web_app.py            Servidor web local y API
app.py                Interfaz Tkinter
menu.py               Menú de terminal
config.py             Rutas y parámetros
media_runtime.py      Inicialización de MediaPipe
storage.py            Validación de datos y modelos
image_io.py           Lectura validada de imágenes
detector.py           Detección y recorte facial
1_setup.py            Descarga de modelos
2_collect_faces.py    Captura y preparación de recortes
3_train.py            Entrenamiento LBPH
4_recognize.py        Reconocimiento
```

El código se distribuye bajo la licencia MIT. Las dependencias y los modelos descargados conservan sus propios términos y licencias.
