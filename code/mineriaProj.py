import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Titulo y subtitulo de la aplicacion
st.set_page_config(layout="centered")
st.title("Análisis de Vehículos Registrados en Municipios de Jalisco (1980-2020)")

df = pd.read_excel("vehiculos_motor_registrados_por_municipio_jalisco_1980-2020.xlsx", na_values='-') # Cargar el archivo Excel
olddf = df

# Eliminar filas adicionales (mantener solo las primeras 127 filas)
df = df[:128]
df.columns = df.columns.astype(str)     # Tratar las columnas como cadenas para graficarlas
for col in df.columns[1:]:      # Convertir columnas de años con valores numéricos 
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
#df es el dataframe original


st.subheader('Objetivo del negocio') # Subheader del objetivo de negocio
st.write("""El crecimiento vehicular en los municipios de Jalisco a lo largo de 41 años es un indicador clave para evaluar la capacidad de la infraestructura vial y la efectividad de las políticas de movilidad. El objetivo principal es identificar 
         los municipios que están experimentando el mayor crecimiento en el número de vehículos y evaluar su riesgo de saturación vial. Este análisis permitirá a los responsables de la planificación urbana y de la infraestructura diseñar estrategias 
         para evitar congestiones de tráfico y priorizar inversiones en infraestructura vial. Además, se busca entender las posibles causas del crecimiento vehicular para recomendar políticas públicas que mitiguen los riesgos de saturación en municipios clave. """)
st.divider()

st.subheader('Objetivo de minería de datos ') # Subheader del objetivo de la mineria de datos
st.write("""El análisis del crecimiento vehicular en los 125 municipios de Jalisco, considerando 41 años de datos (1980-2020), 
         permitirá identificar patrones y tendencias en el crecimiento del parque vehicular. Utilizando técnicas de minería de 
         datos como clustering y análisis de series temporales, se podrán clasificar los municipios según su riesgo de saturación 
         vehicular. Esta clasificación servirá para diseñar estrategias específicas de desarrollo vial y movilidad urbana, adaptadas 
         a las características de cada grupo de municipios. Por otro lado, podríamos utilizar técnicas de análisis exploratorio y 
         algoritmos de minería de datos, como análisis de tendencias y modelos predictivos para identificar los municipios con mayor 
         crecimiento en el número de vehículos. """)
st.divider()

st.subheader('Origen de los datos/ Dataset ') #Subheader del origen de datos
st.write("""El dataset utilizado proviene del Instituto de Información Estadística y Geográfica del Estado de Jalisco (IIEG) y 
         contiene información sobre los vehículos de motor registrados en circulación por municipio en el estado de Jalisco durante 
         el período comprendido entre el 1 de enero de 1980 y el 31 de diciembre de 2020. Los datos han sido recopilados y publicados 
         por la Dirección de Información Estadística de Gobierno, Seguridad Pública e Impartición de Justicia del estado de Jalisco. 
         El formato en el que los datos fueron obtenidos fue en xlxs (Excel), pero debido al formato, decidimos cambiar el archivo a csv. 

El dataset contiene información detallada de 125 municipios del estado de Jalisco, además del estado en su conjunto. 
         Cada fila representa un municipio o el estado en su conjunto, y las columnas contienen los valores anuales de vehículos 
         registrados. Los datos de crecimiento vehicular a lo largo de los años permiten identificar patrones de crecimiento en 
         diferentes municipios. Esto ayudará a determinar cuáles municipios están experimentando un crecimiento acelerado en su parque 
         vehicular y requieren atención prioritaria. 

Cantidad de registros: \n
1. 125 municipios + 1 registro para el total del estado = 126 registros en total. 
2. Cada registro tiene 41 columnas de años (1980 a 2020) más una columna para el nombre del municipio o estado. 
         
Relación con el problema de negocio: Los datos de crecimiento vehicular permiten identificar cuáles municipios están en riesgo 
de saturación vial, información clave para la planificación de infraestructura y políticas públicas. """)
st.divider()
st.subheader("Visualización de los datos iniciales")
st.dataframe(olddf)  # Mostrar el DataFrame clean
st.divider()

st.subheader('Inconsistencias ')
st.write('Existen registros en blanco, por lo que hay municipios con datos incompletos. Estos datos faltantes pueden afectar la precisión de los análisis. \nAdemás hay registros que son tomados como datos numericos y otros que son tomados como cadenas. ')
st.divider()

st.subheader("Columnas del DataFrame inicial:") 
st.info(olddf.columns.to_list())  # Mostrar las columnas de los anios
st.subheader("Columnas y sus valores nulos:")
null_values_df = pd.DataFrame([olddf.isnull().sum()], columns=olddf.columns)
null_values_df.index = ['Valores nulos']
st.dataframe(null_values_df)
st.divider()

# Sección de filtrado
df_aux = df.dropna(axis=1, how='all') # Eliminar columnas completamente vacias
df_aux = df_aux.drop(0) # Eliminar la fila correspondiente a la suma total Jalisco
#df_aux es el dataframe despues de la limpieza e imputaciones
columns = df_aux.columns  
yearColumns = columns[1:]
st.subheader("Filtrado de datos por municipio (datos crudos):")
municipio = st.selectbox("Selecciona un municipio", df_aux.iloc[:, 0])  # Asumimos que la primera columna tiene los nombres de los municipios
df_filtrado = df_aux[df_aux[columns[0]] == municipio]

st.dataframe(df_filtrado)   # Mostrar los datos filtrados
st.divider()

# Visualización de los datos del municipio seleccionado
dfTranspose = df_filtrado[yearColumns].transpose()     # Transponemos el df para que los anios se conviertan en filas
st.subheader(f"Visualización de los datos de {municipio}")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=dfTranspose, ax=ax)  # Transponemos para que las columnas de anios se conviertan en filas
ax.set_title(f"Vehículos registrados en {municipio} de 1980 a 2020")
ax.set_xlabel("Año")
ax.set_ylabel("Número de vehículos")
ax.set_xticks(range(len(yearColumns)))  # Asegurarse de que se muestren todas las etiquetas
ax.set_xticklabels(yearColumns, rotation=90) 
st.pyplot(fig)
st.divider()


#Limpieza
df_clean = df_aux
df_clean=df_clean.drop(df.index[-1])
df_clean = df_clean.drop(columns=[str(year) for year in range(1980, 1997)])
null_values_df_clean = pd.DataFrame([df_clean.isnull().sum()], columns=df_clean.columns)
null_values_df_clean.index = ['Valores nulos']
st.header('Preparación de los datos ')
st.subheader('Selección y limpieza ')
st.write("""Eliminar columnas irrelevantes o con demasiados valores faltantes, como municipios con datos inexistentes para la mayoría de los años. 
         Decidimos eliminar los registros de todos los municipios en el rango 1980 a 1996, debido a la inconsistencia de los datos y la cantidad exagerada de registros nulos. 
         El mantener esa cantidad de nulos en nuestro dataset podría perjudicar nuestros modelos y generar valores erróneos. """)
st.dataframe(null_values_df_clean)

# Imputaciones
df_clean.fillna(0, inplace=True)
numeric_columns = df_clean.select_dtypes(include=['float64', 'int64'])
# Calcular el cuartil inferior (Q1) de cada columna numérica
q1_values = numeric_columns.quantile(0.25).astype(int)
# Reemplazar los valores 0 por el cuartil inferior (Q1)
df_clean[numeric_columns.columns] = numeric_columns.apply(lambda x: x.replace(0, q1_values[x.name]))
st.subheader('Imputación ')
st.write("""Utilizamos el método de cuartil inferior para completar los valores faltantes en los registros donde solo falten algunos años, 
         ya que utilizar la media no nos pareció lo adecuado debido a las diferencias entre los volúmenes de vehículos registrados entre municipios. """)
null_values_df_clean = pd.DataFrame([df_clean.isnull().sum()], columns=df_clean.columns)
null_values_df_clean.index = ['Valores nulos']
st.dataframe(null_values_df_clean)
st.write(pd.DataFrame([q1_values,], columns=numeric_columns.columns))
st.dataframe(df_clean)

#Integraciones
df_grouped = df_clean.groupby('Estado o Municipio').sum(numeric_only=True)
df_grouped['Total Vehiculos'] = df_grouped.sum(axis=1)
st.subheader('Integración de datos ')
st.write("""En nuestro caso, solamente utilizamos un dataframe principal, en el cual operamos y realizamos funciones de agrupación (groupby) y de normalización con la librería 
         scikit-learn, para facilitar la lectura de la densidad de los valores de los registros. Al aplicar la función de agrupación, logramos consolidar los datos por municipio, 
         obteniendo así una visión más clara del total de vehículos registrados.  """)
st.dataframe(df_grouped)

# Transformaciones
scaler = MinMaxScaler()
df_grouped['Total Vehiculos Normalized'] = scaler.fit_transform(df_grouped[['Total Vehiculos']])
st.subheader('Transformación ')
st.write("""Posteriormente, implementamos la normalización para ajustar los valores de la nueva columna 'Total Vehículos', lo que nos permitió comparar fácilmente las cifras entre los diferentes 
         municipios y observar tendencias en la densidad de vehículos registrados. Esta metodología nos permitió simplificar el análisis y mejorar la interpretabilidad de los resultados. """)
st.dataframe(df_grouped)
st.divider()

# Filtrado 2.0
columns_final = df_clean.columns
yearColumns_final = columns_final[1:]
st.subheader("Filtrado de datos por municipio (datos procesados):")
municipio_final = st.selectbox("Selecciona un municipio", df_clean.iloc[:, 0], key="municipio_final")  # Asumimos que la primera columna tiene los nombres de los municipios
df_filtrado_final = df_clean[df_clean[columns_final[0]] == municipio_final]

st.dataframe(df_filtrado_final)   # Mostrar los datos filtrados
st.divider()

# Visualización de los datos del municipio seleccionado
dfTranspose_final = df_filtrado_final[yearColumns_final].transpose()     # Transponemos el df para que los anios se conviertan en filas
st.subheader(f"Visualización de los datos de {municipio_final}")
fig2, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=dfTranspose_final, ax=ax)  # Transponemos para que las columnas de anios se conviertan en filas
ax.set_title(f"Vehículos registrados en {municipio_final} de 1997 a 2020")
ax.set_xlabel("Año")
ax.set_ylabel("Número de vehículos")
ax.set_xticks(range(len(yearColumns_final)))  # Asegurarse de que se muestren todas las etiquetas
ax.set_xticklabels(yearColumns_final, rotation=90) 
st.pyplot(fig2)
st.divider()

st.header("Número de celdas antes y después de la limpieza")
st.subheader('Antes')
st.write('Dimensiones: ' + str(olddf.shape))
st.write("Número de celdas: "+ str(olddf.size))
st.subheader('Después')
st.write('Dimensiones: ' + str(df_clean.shape) )
st.write("Número de celdas: "+ str(df_clean.size))
st.divider()

#Seleccion de modelos
st.header('Seleccion de modelos')
opcion=st.selectbox("Matriz de correlacion:", ["Grafica", "Tabla"], key="opcion")
if opcion=="Tabla":
    st.dataframe(df_clean.corr(numeric_only=True))
else:
    corrMatrix= df_clean.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de correlación del dataset')
    st.pyplot(plt)
st.write("Al realizar una exploración en la matriz de correlación, observamos que las correlaciones entre las variables son muy altas. Sin embargo, esto se debe a que simplemente se muestra como el número de vehículos va aumentando año con año.")           #MATRIZ CORRELACION JUSTIFICAR
st.divider()

#Aplicar regression lineal
st.header('Aplicaremos regression lineal para predecir crecimiento vehicular por municipio')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# Convertir anios a numericos
for col in df_clean.columns[1:]:
    df_clean[col] = df_clean[col].astype(str).str.replace(',', '', regex=False).astype(float)

municipio_final2 = st.selectbox("Selecciona un municipio", df_clean["Estado o Municipio"])
anio = st.number_input("Año para predecir (hasta 2300)", min_value=1997, max_value=2300, step=1, format="%d")

fila_municipio = df_clean[df_clean["Estado o Municipio"] == municipio_final2].iloc[0, 1:]# Filtrar datos del municipio seleccionado
anios = np.array([int(a) for a in fila_municipio.index])  # Convertir los anios a enteros
valores = np.array(fila_municipio.values)  # Valores de vehiculos registrados

X = anios.reshape(-1, 1) # Transformar los datos en matriz bidimensional
y = valores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Dividir los datos en entrenamiento y prueba

modelo = LinearRegression()# Crear y entrenar el modelo de regresion lineal
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)# Prediccion
nuevo_valor = modelo.predict(np.array([[anio]]))[0]
mse = mean_squared_error(y_test, y_pred)    # Metricas del modelo
r2 = modelo.score(X_test, y_test)

st.subheader(f"Resultados para el municipio: {municipio_final2}")
st.write(f"Predicción para el año {anio}: **{nuevo_valor:,.2f} vehículos**")
if(anio in anios):
    st.write(f"Datos reales: ", str(fila_municipio[str(anio)]))
st.write(f"Error cuadrático medio (MSE): {mse:,.2f}")
st.write(f"R cuadrada (R²): {r2:.2f}")

fig, ax = plt.subplots()
plt.scatter(anios, valores, color="blue", label="Datos Reales")
plt.plot(anios, modelo.predict(X), color="red", label="Modelo de Regresión")
plt.axvline(x=anio, color="green", linestyle="--", label=f"Predicción para {anio}")
plt.scatter(anio, nuevo_valor, color="orange", label=f"Predicción: {nuevo_valor:,.2f}")
plt.xlabel("Año")
plt.ylabel("Vehículos registrados")
plt.title(f"Regresión lineal para {municipio_final2}")
plt.legend()
st.pyplot(fig)


# Parte de la ninia nerda
st.divider()
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#Modelo K-means
st.header("Modelo de K-Means")

#Se seleccionan entre 5 y 5 para no seleccionar todos los años y solo tomar los más significativos
numerical_columns = ['1997', '2002', '2007', '2012', '2017', '2020']

#Normalización de los datos usando MinMaxScaler
scaler = MinMaxScaler()
df_clean[numerical_columns] = scaler.fit_transform(df_clean[numerical_columns])

#Mostrar datos normalizados
st.subheader("Datos Normalizados")
st.dataframe(df_clean[numerical_columns].head())

# Determinación del valor óptimo de K (Gráfico de Codo)
st.title("Determinación del Valor Óptimo de K (Gráfico de Codo)")

# Crear un rango de valores de K
Nc = range(1, 20)

# Cálculo de la inercia para cada valor de K
X = df_clean[numerical_columns].values  #datos ya normalizados
kmeans = [KMeans(n_clusters=i, random_state=42) for i in Nc]
inertia = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

# Crear el gráfico de codo
fig, ax = plt.subplots()
ax.plot(Nc, inertia, marker='o', color='green')
ax.set_xlabel('Número de Clústers')
ax.set_ylabel('Inercia')
ax.set_title('Gráfico de Codo')

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
st.write("Podemos observar que el punto que nos interesa (el codo), se encuentra en aproximadamente 3.")

#Aplicamos K-means con 3 clústers
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
centroids = kmeans.cluster_centers_

# Mostrar los valores de los centroides
st.subheader("Valores de los Centroides")
st.write(centroids)

# Etiquetas de los clústers
labels = kmeans.predict(X)

# Clustering de municipios usando K-means (Visualización 3D)
st.header('Clustering de municipios usando K-means')

# Seleccionamos tres columnas para el gráfico 3D
vehicle_data = df_clean[['1997', '2010', '2020']]  # Seleccionamos tres columnas de datos numéricos
vehicle_data_normalized = scaler.fit_transform(vehicle_data)  # Normalizamos los datos

# Selección de número de clusters
n_clusters = st.slider("Selecciona el número de clusters", min_value=2, max_value=10, value=3)

# Aplicamos el modelo K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(vehicle_data_normalized)

# Determinación de los centroides
centroids = kmeans.cluster_centers_

# Colores para cada cluster
colores = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink']
asignar = [colores[label] for label in labels]

# Crear el gráfico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dibujar los puntos de datos
ax.scatter(vehicle_data_normalized[:, 0], vehicle_data_normalized[:, 1], vehicle_data_normalized[:, 2], 
           c=asignar, s=60, label='Municipios')

# Dibujar los centroides de los clusters
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='black', s=300, label='Centroides')

# Etiquetas y titulo
ax.set_xlabel('Vehículos en 1997')
ax.set_ylabel('Vehículos en 2010')
ax.set_zlabel('Vehículos en 2020')
ax.set_title('Clustering de Municipios - 3D')
ax.legend()

# Mostrar el gráfico
st.pyplot(fig)

# Mostrar las asignaciones de cluster
st.subheader('Municipios por Cluster')
df_clean['Cluster'] = labels
st.write(df_clean[['Estado o Municipio', 'Cluster']])

