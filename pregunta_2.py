import pandas as pd

data = pd.read_csv('matriz_TF_IDF_agrupada.csv')

data.drop(['Pokemon', 'grupos'], inplace=True)
