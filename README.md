# Machine Learning - Tutorial Interactivo

Aplicación web interactiva creada con Streamlit para enseñar conceptos de Machine Learning de forma práctica y visual.

## Estructura del Proyecto

```
ml-uc3m-streamlit/
├── app.py                              # Aplicación principal
├── requirements.txt                     # Dependencias
├── README.md                           # Este archivo
└── capitulos/                          # Módulos de cada capítulo
    ├── 01_introduccion/
    │   └── introduccion.py
    ├── 02_knn/
    │   └── knn.py
    ├── 03_arboles/
    │   └── arboles.py
    ├── 04_evaluaciones/
    │   └── evaluaciones.py
    ├── 05_metricas/
    │   └── metricas.py
    ├── 06_ajuste_hiperparametros/
    │   └── ajuste_hiperparametros.py
    └── 07_preproceso/
        └── preproceso.py              # ✅ Con ejemplos interactivos completos
```

## Capítulos Disponibles

1. **Introducción** - Conceptos básicos de ML
2. **K-Nearest Neighbors (KNN)** - Algoritmo de clasificación por vecinos
3. **Árboles de Decisión** - Modelos basados en árboles
4. **Evaluación de Modelos** - Cómo evaluar el rendimiento
5. **Métricas de Rendimiento** - Accuracy, Precision, Recall, F1, etc.
6. **Ajuste de Hiperparámetros** - Grid Search, Random Search, etc.
7. **Preprocesamiento de Datos** - ✅ **Implementado con ejemplos interactivos**
   - Imputación de valores faltantes (Media, Mediana, KNN, MICE)
   - Escalado de datos (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)

## Instalación

1. Clona el repositorio o navega al directorio del proyecto

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecuta la aplicación con:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Características

- **Interfaz interactiva** con Streamlit
- **Visualizaciones en tiempo real** con Matplotlib y Seaborn
- **Ejemplos prácticos** en 2D para mejor comprensión
- **Parámetros ajustables** para experimentar con diferentes configuraciones
- **Estructura modular** - cada capítulo es independiente

## Capítulo de Preprocesamiento (Implementado)

El capítulo 7 incluye ejemplos interactivos completos:

### Imputación de Valores Faltantes
- Visualización de datos antes y después de la imputación
- Múltiples algoritmos: Media, Mediana, Moda, KNN, Iterativo (MICE)
- Métricas de error (MAE, RMSE)
- Comparación visual en 2D

### Escalado de Datos
- Comparación de diferentes métodos de escalado
- Visualización de transformaciones en 2D
- Estadísticas comparativas
- Histogramas de distribución
- Explicación detallada de cuándo usar cada método

## Próximos Pasos

Los capítulos 1-6 están inicializados con estructura básica y están listos para ser desarrollados con contenido interactivo similar al capítulo de preprocesamiento.

## Requisitos

- Python 3.8+
- Streamlit 1.31.0
- NumPy 1.26.3
- Pandas 2.2.0
- Matplotlib 3.8.2
- Seaborn 0.13.1
- scikit-learn 1.4.0

## Licencia

Este proyecto es material educativo para el curso de Machine Learning de UC3M.