# 🪐 Clasificación de habitabilidad basado en temperatura de equilibrio de exoplanetas
**Analisis de habitabilidad termica utilizando modelos de:**

- SVM (Support Vector Machine)
- Random Forest
- Boosting
- Red Neuronal

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## Descripción

Este proyecto utiliza los modelos mencionados para clasificar y predecir la **habitabilidad** de exoplanetas y determinar cuáles podrían ser potencialmente habitables según criterios térmicos 273 - 373 K (rango de temperatura permisible).

**Dataset**: NASA Exoplanet Archive (Dataset ya limpio) 
**Objetivo**: realizar una comparación de modelos para clasificación de habitabilidad.

Recursos

| Dataset |[NASA_exoplanets (ya imputada)](dataset_final_exoplanetas.csv) |
|---|---|
| **Notebook** | [.ipynb](A2_2_LDA_Arboles.ipynb) |

---


## Metodología

- Definición de la variable de salida

- División del dataset en entrenamiento y prueba, conservando la proporción de clases.

- Construcción del modelos , seleccionando variables acordes a sus supuestos.

- Visualización e interpretación de metricas de desempeño, F1- Score, OOB-Score, accuracy, etc.

- Comparación de modelos de clasificación.

- Conclusiones acerca de los modelos.

---

## Procedimiento

[Definir variable de salida](variable_salida.md)

[Split de datos](split.md)

[Modelo Random Forest](Modelo_RF.md)

[Modelo Boosting](Modelo_boosting.md)

[Modelo SVM](modelo_SVM.md)

[Modelo Red Neurona / Conclusionesl](modelo_RN.md)


## Referencias

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- Kopparapu et al. (2013). *Habitable Zones Around Main-Sequence Stars*
- Kasting et al. (1993). *Habitable Zones around Main Sequence Stars*
---



**By** 

*Juan Angel Candelaria Rodriguez*

Universidad de Monterrey | Inteligencia Artificial  


