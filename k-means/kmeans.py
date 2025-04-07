import sys

from sklearn.cluster import KMeans
from cleaning import load_normalized_dataset, load_clean_dataset
import pandas as pd

# Cargar los datos limpios y normalizados
df_original = load_clean_dataset()  # Datos originales (sin la clase)
df_normalized, scaler = load_normalized_dataset()  # Datos normalizados

# Aplicar KMeans con 3 clusters sobre los datos normalizados
k = 7
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_normalized)

# Agregar la columna 'Cluster' al DataFrame original
df_original['Cluster'] = clusters

# Mostrar los centroides en su escala original
centroides_originales = scaler.inverse_transform(kmeans.cluster_centers_)

# Crear un DataFrame con los centroides, con los nombres de los atributos como columnas
centroides_df = pd.DataFrame(centroides_originales, columns=df_original.columns[:-1])

# Mostrar los centroides con los nombres de los atributos
print("Centroides en escala original con los nombres de los atributos:")
print(centroides_df)
