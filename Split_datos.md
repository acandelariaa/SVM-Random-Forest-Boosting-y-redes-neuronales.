# Split train y test

En este apartado dividiremos los datos de entrenamiento en train y test y brevemente 
exploraremos la naturaleza de los datos, donde confirmaremos que efectivamente estan listos para ser trabajados.


### Dividir datos

Dividiremos los datos con una proporcion de 50/50, asi mismo utilizaremos el comando stratify para 
asegurar que los datos de train y test tengan la misma proporción.


>Python Code


```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['pl_name', 'habitable', 'pl_eqt'])
y = df['habitable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=42, stratify=y
)

print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")
print("\nDistribución en train:")
print(y_train.value_counts())
print("\nDistribución en test:")
print(y_test.value_counts())
```


>Output


```text
Registros de entrenamiento: 2283
Registros de prueba: 2283

Distribución en train:
habitable
0    2168
1     115
Name: count, dtype: int64

Distribución en test:
habitable
0    2169
1     114
Name: count, dtype: int64

```


Antes de eso vayamos a confirmar que el conjunto de datos esta listo para usar, para eso utilicemos varias tecnicas, 
como el metodo de tukey para verificar los outliers y el porcentaje que estos conforman en el dataset, asi mismo 
verifiquemos si tenemos valores nulos o huecos, esto con el fin 
de satisfacer lo que viene estipulado en las instruciones de la tarea.


### Verificar limpieza de datos

Para verificar la limpieza de los datos, utilizaremos el metodo de tukey, donde definiremos nuestro rango 
intercuartil Q3-Q1, donde los valores que caigan fuera de ese rango, seran considerados como outliers. Así mismo verificaremos si hay huecos en el dataset.

>Python Code


```python
# ============================================================
# REVISIÓN FINAL DE LA BASE DE DATOS
# ============================================================
import pandas as pd
import numpy as np
# --- 1. Valores nulos ---
print("=" * 50)
print("VALORES NULOS")
print("=" * 50)
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
resumen_nulos = pd.DataFrame({'Nulos': nulos, 'Porcentaje (%)': pct_nulos})
resumen_nulos = resumen_nulos[resumen_nulos['Nulos'] > 0].sort_values('Porcentaje (%)', ascending=False)
print(resumen_nulos)
print(f"\nTotal columnas con nulos: {len(resumen_nulos)}")

# --- 2. Outliers por método de Tukey (IQR) ---
print("\n" + "=" * 50)
print("OUTLIERS - MÉTODO DE TUKEY (IQR)")
print("=" * 50)

columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
resumen_outliers = []

for col in columnas_numericas:
    serie = df[col].dropna()
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    n_outliers = ((serie < limite_inf) | (serie > limite_sup)).sum()
    pct = round(n_outliers / len(serie) * 100, 2)

    resumen_outliers.append({
        'Variable': col,
        'Q1': round(Q1, 4),
        'Q3': round(Q3, 4),
        'IQR': round(IQR, 4),
        'Límite inferior': round(limite_inf, 4),
        'Límite superior': round(limite_sup, 4),
        'Outliers': n_outliers,
        'Porcentaje (%)': pct
    })

df_outliers = pd.DataFrame(resumen_outliers).sort_values('Porcentaje (%)', ascending=False)
display(df_outliers)

# Resumen global
total_outliers = df_outliers['Outliers'].sum()
print(f"\nTotal de valores atípicos detectados: {total_outliers}")
print(f"Porcentaje promedio de outliers por variable: {df_outliers['Porcentaje (%)'].mean():.2f}%")
```

>Output


```text
==================================================
VALORES NULOS
==================================================
Empty DataFrame
Columns: [Nulos, Porcentaje (%)]
Index: []

Total columnas con nulos: 0

==================================================
OUTLIERS - MÉTODO DE TUKEY (IQR)
==================================================

| Variable  | Q1        | Q3       | IQR      | Límite inferior | Límite superior | Outliers | Porcentaje (%) |
|-----------|-----------|----------|----------|-----------------|-----------------|----------|----------------|
| pl_insol  | 24.1035   | 397.099  | 372.9955 | -535.3897       | 956.5922        | 521      | 11.41          |
| st_teff   | 4999.2500 | 5927.750 | 928.5000 | 3606.5000       | 7320.5000       | 284      | 6.22           |
| habitable | 0.0000    | 0.000    | 0.0000   | 0.0000          | 0.0000          | 229      | 5.02           |
| st_rad    | 0.7700    | 1.190    | 0.4200   | 0.1400          | 1.8200          | 190      | 4.16           |
| pl_eqt    | 568.0000  | 1165.000 | 597.0000 | -327.5000       | 2060.5000       | 100      | 2.19           |

Total de valores atípicos detectados: 1324
Porcentaje promedio de outliers por variable: 5.80%
```


Bien, parece que no hay valores nulos, esto es debido a que la base ya habia sido imputada con KNN, pero parece que 
siguen habiendo valores atipicos, de modo que podriamos tratar de llenarlos con KNN de la misma manera.


### Reemplazar outliers con KNN

Con el fin de no tener tantos datos atipicos, usaremos KKN imputer, en donde primero se
identificaran los outliers, se cambiaran a valores NaN y luego se llenaran esos datos con este metodo.

>Python Code



```python
from sklearn.impute import KNNImputer
import numpy as np

columnas_modelo = ['pl_insol', 'st_teff', 'st_rad']

# --- 1. Convertir outliers a NaN usando Tukey ---
for col in columnas_modelo:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    # Reemplazar outliers con NaN
    df[col] = np.where(
        (df[col] < limite_inf) | (df[col] > limite_sup),
        np.nan,
        df[col]
    )
    print(f"{col}: NaN después de marcar outliers: {df[col].isna().sum()}")

# --- 2. Imputar los NaN con KNN ---
imputer = KNNImputer(n_neighbors=5)
df[columnas_modelo] = imputer.fit_transform(df[columnas_modelo])

# --- 3. Verificar ---
print("\nOutliers restantes después de imputación:")
for col in columnas_modelo:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col}: {outliers}")

```

>Output


```
pl_insol: NaN después de marcar outliers: 521
st_teff: NaN después de marcar outliers: 284
st_rad: NaN después de marcar outliers: 190

Outliers restantes después de imputación:
  pl_insol: 241
  st_teff: 14
  st_rad: 58

```


\Muy interesante lo que estamos viendo, aun despues de haber aplicado el metodo de KNN para remplazar los 
valores atipicos, parece que siguen habiendo valores extremos, esto podria sugerir que la naturaleza de algunos
datos es real y no es un error de captura, por lo que para fines practicos lo dejaremos asi.

-----
Siguiente pagina
