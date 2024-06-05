# Importacion de librerias

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

# Importaciond de data

data = pd.read_csv('smogon.csv')

print(data.columns)

# Matriz TF-IDF

# Convertir toda la columan a minusculas

# Convertir todos los elementos de la columna 'Columna' a minúsculas
data.loc[:, 'moves'] = data['moves'].str.lower()

# usamos unigramas y bigramas y el conjunto predeterminaod de stop words en ingles de sklearn
n_gramas = (1,2) 

tfidf = TfidfVectorizer(ngram_range=n_gramas, stop_words='english')

# Generamos la matriz TF-IDF
matrix_TFIDF = tfidf.fit_transform(data['moves'])

# Mostramos el numero total de tokens
print(f"Numero total de tokens: {len(tfidf.vocabulary_)}")

# Mostramos los tokens ordenados
tokens = sorted(tfidf.vocabulary_)

print(f"Tokens: {tokens}")

# Mostramos la matriz TF-IDF con sus cabeceras
matrix_TFIDF_limpia = pd.DataFrame(matrix_TFIDF.toarray(), columns=tokens)

print(matrix_TFIDF_limpia)

# Agrupamos las filas del nuevo DataFrame 
k = 18

km = KMeans(n_clusters=k, n_init=10)
grupos = km.fit_predict(matrix_TFIDF_limpia)

# Juntar los grupos con la matriz TF-IDF y los nombres para poder analizar
matrix_TFIDF_limpia['grupos'] = grupos
matrix_TFIDF_limpia['Pokemon'] = data['Pokemon']

# Guardamos en un CSV la matriz obviando los indices
matrix_TFIDF_limpia.to_csv('matriz_TF_IDF_agrupada.csv',index=False)