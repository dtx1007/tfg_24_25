# TFG 2024-2025: Detección de Malware en Android

> Repositorio principal de desarrollo para el TFG sobre Detección de malware en Android mediante técnicas de Inteligencia Artificial y análisis estático.

## 📋 Índice

- [🔎 ¿Qué es?](#-qué-es)
- [⌛ Historia](#-historia)
- [🐋 Despliegue](#-despliegue)
- [🔧 Desarrollo](#-desarrollo)
- [📜 Licencia](#-licencia)
- [🌟 Menciones y agradecimientos](#-menciones-y-agradecimientos)

## 🔎 ¿Qué es?

Este repositorio contiene todo el trabajo realizado durante el Trabajo de Fin de Grado (TFG) centrado en la detección de *malware* en aplicaciones Android (`.apk`). El proyecto explora la viabilidad de utilizar técnicas de análisis estático e Inteligencia Artificial, concretamente redes neuronales profundas, para clasificar aplicaciones como benignas o maliciosas.

El propósito principal de este trabajo ha sido la **investigación y el desarrollo**. Por tanto, este repositorio alberga no solo la aplicación final, sino todo el código fuente, los cuadernos de experimentación (`Jupyter Notebooks`), el proceso de creación del *dataset*, el entrenamiento de los modelos y la documentación asociada.

Para una inmersión profunda en la metodología, el estado del arte, el análisis de resultados y las conclusiones del proyecto, se recomienda encarecidamente consultar la [**memoria completa del TFG**](./doc/memoria.pdf), disponible en este mismo repositorio.

## ⌛ Historia

El desarrollo de este proyecto fue un viaje iterativo que partió de una pregunta fundamental: ¿es posible detectar *malware* de forma fiable sin ejecutarlo?

1. **Fase de Investigación:** El proyecto comenzó con una exploración del estado del arte para validar la idea. Se concluyó que el análisis estático en Android mediante IA era un campo prometedor pero con desafíos, especialmente en la reproducibilidad de los resultados.

2. **Fase de Prototipado:** Se construyó un primer modelo utilizando el conocido *dataset* Drebin. Esta prueba de concepto fue un éxito y demostró que una red neuronal podía aprender patrones de malicia, pero también evidenció la necesidad de crear un *pipeline* de datos propio para poder aplicar el modelo a nuevas aplicaciones.

3. **Fase de Desarrollo:** El núcleo del trabajo se centró en la creación de un *dataset* a medida a partir de miles de APKs obtenidas del repositorio AndroZoo. Se implementó un *pipeline* completo con Androguard para la extracción de características, y se diseñó y optimizó un modelo de red neuronal final.

4. **Análisis y Demostración:** Finalmente, se evaluó el rendimiento del modelo, se comparó con otros algoritmos clásicos y se analizó su interpretabilidad. El resultado de todo este proceso es la aplicación web de demostración que se puede desplegar desde este repositorio.

## 🐋 Despliegue

Este repositorio contiene todo el código necesario para desplegar la aplicación. Sin embargo, para un despliegue rápido y sencillo, se recomienda utilizar el **repositorio de despliegue**, que ha sido creado para este fin:

➡️ **Repositorio de Despliegue:** [https://github.com/dtx1007/streamlit_malware_detection_app](https://github.com/dtx1007/streamlit_malware_detection_app)

A continuación se explicará el proceso de despliegue basándose en el repositorio anteriomente mencionado.

**Requisitos previos:**

- [**Git**](https://git-scm.com/downloads)
- [**Git LFS**](https://git-lfs.com/) (esencial para descargar los modelos)
- [**Docker**](https://www.docker.com/)
- [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (opcional; solo si se quiere ejecutar el contenedor con soporte para CUDA)

**Pasos:**

1. **Clonar el repositorio de despliegue**

    Es crucial usar `git lfs pull` tras clonar el repositorio para descargar los modelos de machine learning; debido a tu tamaño se gestionan mediante Git LFS.

    ```sh
    # Clona el repositorio
    git clone https://github.com/dtx1007/streamlit_malware_detection_app
    cd streamlit_malware_detection_app

    # Descarga los archivos LFS
    git lfs pull
    ```

2. **Construir la imagen Docker**

    La imagen se puede buildear para ejecutarse en CPU o en GPU si se dispone de una tarjeta NVIDIA compatible con CUDA.

    ```sh
    # Build para CPU
    docker build -t streamlit-malware-app:cpu .

    # Build con soporte para CUDA (GPU)
    docker build --build-arg BUILD_TYPE=gpu -t streamlit-malware-app:gpu .
    ```

3. **Ejecutar el contenedor**

    Una vez construida la imagen, inicie el contenedor. La aplicación estará disponible en `http://localhost:8501`.

    ```sh
    # Ejecutar el contenedor de CPU
    docker run -p 8501:8501 streamlit-malware-app:cpu

    # Ejecutar el contenedor de GPU
    docker run --gpus all -p 8501:8501 streamlit-malware-app:gpu
    ```

## 🔧 Desarrollo

Este apartado está destinado a desarrolladores que deseen contribuir al proyecto, experimentar con el código o entrenar sus propios modelos.

**Requisitos previos:**

- **Python 3.11+**
- **Poetry** para la gestión de dependencias y entornos virtuales.

**Configuración del entorno:**

1. **Clona el repositorio** (asegúrate de incluir los archivos de LFS como se indica en la sección de despliegue).

2. **Instala las dependencias**

    Poetry se encargará de crear un entorno virtual y de instalar todas las librerías necesarias definidas en el archivo `pyproject.toml`. Al igual que para el despliegue, es necesario elegir si se desea tener soporte para CUDA.

    ```sh
    # Instalación base
    poetry install --extras=cpu

    # Instalación con soporte para CUDA (GPU)
    poetry install --extras=gpu
    ```

3. **Activa el entorno virtual**

    ```sh
    poetry shell

    # Alternativa si 'poetry shell' no está disponible:
    # Para bash / zsh
    eval $(poetry env activate)
    # Para PowerShell
    Invoke-Expression (poetry env activate)
    ```

    Una vez dentro del *shell*, podrás ejecutar los *scripts* y *notebooks* del proyecto.

**Estructura del Repositorio:**

Para facilitar la navegación, aquí se presenta una descripción de los directorios más importantes:

<!-- TODO: Subir dataset y alguna de las apks -->

```txt
.
├── apks/               # Artefactos del dataset propio (no se incluye)
├── dataset/            # Datasets empleados durante el entrenamiento
├── doc/                # Contiene la memoria completa del TFG en formato LaTeX.
├── model_artifacts/    # Modelos entrenados, vocabularios y otros artefactos generados.
├── plots/              # Gráficas generadas de los resultados de entrenar los modelos y su interpretabilidad.
└── src/                # Código fuente principal del proyecto.
    ├── app/            # Código fuente de la aplicación web con Streamlit.
    ├── notebooks/      # Jupyter Notebooks para exploración y análisis.
    ├── prototypes/     # Módulos con las arquitecturas de los modelos y su I/O.
    └── utils/          # Scripts de utilidad (extracción de características, preprocesamiento, etc.).
```

## 📜 Licencia

TODO: Elegir licencia

## 🌟 Menciones y agradecimientos

Proyecto realizado por:

- David Cezar Toderas ([dtx1007](https://github.com/dtx1007))

Proyecto coordinado por:

- Álvar Arnaiz González ([alvarag](https://github.com/alvarag))

Agradecer también a la [Universidad de Burgos](https://www.ubu.es/) por ofrecer la posibilidad de realizar este proyecto.

## ✒️

> "*You can't defend. You can't prevent. The only thing you can do is detect and respond.*" – Bruce Schneier
