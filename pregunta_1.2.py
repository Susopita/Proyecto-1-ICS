# Importamos las librerias necesarias
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Cargamos la data de la Parte 1
data = pd.read_csv('matriz_TF_IDF_agrupada.csv')

# Descartamos la columna 'grupos' y 'Pokemon'
data.drop(columns=['Pokemon', 'grupos'], inplace=True)

# Mostramos el DataFrame
print(data.head(10))

# En el Parte 1 a la hora de guardar el CSV, se omitio los indices, por lo que ya no es necesario hacerlo en este paso

# creamos nuestro objeto PCA
pca = PCA()

# Establecemos un porcentaje de precision
ratio = 0.95 

# Obtenemos la variacion acumulada
variacion_acumulada = np.cumsum(pca.fit(data).explained_variance_ratio_)

# Calculamos los componentes "ideales" a nuestro ratio
componentes_ideal = np.argmax(variacion_acumulada >= ratio) + 1

# Asignamos los componentes 
pca = PCA(n_components=componentes_ideal)

# Aplicamos PCA
data_PCA = pca.fit_transform(data)

# Mostramos el numero de filas y columnas de la data real
print(data.shape)

# Mostraremos el numero de filas y columnas de la matriz de PCA
print(data_PCA.shape)

# Generamos las cabeceras con una lista de compresion
cabeceras = [f"PC{i}" for i in range(1, componentes_ideal+1)]

# Agreguamos y generamos un DataFrame nuevo con las cabeceras
data_PCA_con_Headers = pd.DataFrame(data=data_PCA, columns=cabeceras)

# Mostramos el nuevo DataFrame
print(data_PCA_con_Headers.head(10))

# Agrupa la data con kmenas XD sorry ahi lo dejo pa ti (borra luego esta linea)