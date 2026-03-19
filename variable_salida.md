### Definir variable de salida.

En este apartado exploraremos brevemente el comportamiento de los datos que tenemos en el datastet.

>Nota: el dataset esta ya limpio para trabajar, sin embargo se verifico su estado con el fin de confirmar que estamos
>trabajando con una base de datos lista para la construcción de modelos.



### Carga de datos


>Python Code



```python
from google.colab import drive
drive.mount('/content/drive')

# importar datos
import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/dataset_final_exoplanetas.csv')
df.head(5)

```

>Output

| pl_name   | pl_eqt | pl_insol   | st_teff | st_rad   |
|-----------|--------|------------|---------|----------|
| 11 Com b  | 1700.0 | 792.57818  | 4060.0  | 1.313160 |
| 11 UMi b  | 1450.0 | 372.74340  | 3100.0  | 0.840000 |
| 14 And b  | 1600.0 | 513.74032  | 3530.0  | 0.366450 |
| 14 Her b  | 2369.0 | 427.54252  | 2943.0  | 0.289776 |
| 16 Cyg B b| 2677.0 | 5262.59620 | 7244.6  | 1.976258 |

Ahora, parece que la base de datos no tiene una variable con la cual clasificar, por lo que crearemos una 
variable binaria llamada `habitable`, donde tendremos 1=Si y 0 = No; esto basado en el rango de temperatura de [273 - 373 K]


### Crear variable binaria de habitabilidad


>Python Code



```python
# crear variable segun rango de temperatura
df['habitable'] = ((df['pl_eqt'] >= 273) & (df['pl_eqt'] <= 373)).astype(int)
# imprimir conteo de planetas habitables y no habitables
print(df['habitable'].value_counts())
print(df['habitable'].dtype)

```

>Output



```text
habitable
0    4337
1     229
Name: count, dtype: int64
int64
```

Bien, como podemos apreciar, hay desbalance en las clases, ya que tenemos muchos mas planetas los cuales no son habitables, 
esto podria generar sesgos si no lo tratamos con cuidado.

---

[Siguiente Pagina](Split_datos.md)
