# Importo las librerias necesarias
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Cargamos e CSV smogon
data = pd.read_csv('smogon.csv', usecols=['Pokemon','moves'])

# Con la ayuda de gpt4o, obtengo la lista de todos los tipos pokemon
types = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy"
]

# Usando lista de compresion, reprocesamos la columna «moves» de una forma pythonica
for i, moves in enumerate(data['moves']):
    data.loc[i, 'moves'] = " ".join([tipo for tipo in types if tipo in moves.lower() for _ in range(moves.lower().count(tipo))])

# Procesamos la columna «moves» con TF-IDF 
# Usamos unigramas
n_gramas = (1,1) 

tf_idf = TfidfVectorizer(ngram_range=n_gramas)

matriz_TF_IDF = tf_idf.fit_transform(data['moves'])

# Mostramos los tokens ordenados
tokens = sorted(tf_idf.vocabulary_)
print(f"Tokens:\n{tokens}")

# Indicamos la cantidad de tokens en total
print(f"Número de Tokens: {len(tokens)}")

matriz_TF_IDF = pd.DataFrame(matriz_TF_IDF.toarray(), columns=tokens)

# Inicializamos K-Means
k = 18

km = KMeans(n_clusters=k, n_init=10)
grupos = km.fit_predict(matriz_TF_IDF)

matriz_TF_IDF['grupos'] = grupos
matriz_TF_IDF['Pokemon'] = data['Pokemon']

matriz_TF_IDF.to_csv('moves.csv', index=False)