import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


# Cargar el archivo CSV
jal_df = pd.read_excel("vehiculos_motor_registrados_por_municipio_jalisco_1980-2020.xlsx", na_values='-') # Cargar el archivo Excel

# Eliminar filas adicionales (mantener solo las primeras 128 filas)
df = jal_df[:127]


#Eliminar columna Jalisco (suma de todos los vehiculos
df_clean = df.drop(0)
df_clean.columns = df_clean.columns.astype(str)     # Tratar las columnas como cadenas para graficarlas

# Mostrar el DataFrame limpio sin columnas completamente vacías
print(df_clean)

# Justificación de eliminación de columnas de 1980 a 1996# Conteo de valores nulos después de limpiar las columnas vacías
null_clean = df_clean.isnull().sum()

# Eliminar columnas correspondientes a los años 1980 a 1996
df_final = df_clean.drop(columns=[str(year) for year in range(1980, 1997)])

# Mostrar conteo de valores nulos tras la eliminación de las columnas de 1980 a 1996
null_count_final = df_final.isnull().sum()

df_final.fillna(0, inplace=True)
# Mostrar el DataFrame final
print(df_final)
for year in range(1997, 2021):
    column_name = str(year) 
    df_final[column_name] = df_final[column_name].replace({',': ''}, regex=True).astype(int)

#Sacar el cuartil inferior por año para imputarlo en los valores nulos

# Filtrar solo las columnas numéricas para evitar problemas con las columnas 'Unnamed'
# Identificar las columnas numéricas
numeric_columns = df_final.select_dtypes(include=['float64', 'int64'])

# Calcular el cuartil inferior (Q1) de cada columna numérica
q1_values = numeric_columns.quantile(0.25).astype(int)



# Reemplazar los valores 0 por el cuartil inferior (Q1)
df_final[numeric_columns.columns] = numeric_columns.apply(lambda x: x.replace(0, q1_values[x.name]))


# Mostrar el conteo de valores nulos después de la imputación
print(df_final)


# Verificar los tipos de datos después de la conversión
print(df_final.dtypes)

# Agrupar por 'Estado o Municipio' y sumar los valores de las columnas numéricas
df_grouped = df_final.groupby('Estado o Municipio').sum(numeric_only=True)

# Agregar una nueva columna 'Total Vehiculos' que sea la suma total de los vehículos registrados por cada municipio
df_grouped['Total Vehiculos'] = df_grouped.sum(axis=1)

# Normalizar la columna 'Total Vehiculos' usando MinMaxScaler
scaler = MinMaxScaler()
df_grouped['Total Vehiculos Normalized'] = scaler.fit_transform(df_grouped[['Total Vehiculos']])

# Mostrar el DataFrame con la nueva columna normalizada
print(df_grouped.head())