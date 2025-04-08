import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

# === RUTA DEL ARCHIVO ===
ruta_arff = r"C:\Users\maric\Documentos\Mineria de datos\DryBeanClustering\Dry_Bean_Dataset.arff"

# === FUNCIONES ===

def load_clean_dataset(ruta=ruta_arff):
    data, meta = arff.loadarff(ruta)
    df = pd.DataFrame(data)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    return df

def load_normalized_dataset(ruta=ruta_arff):
    df = load_clean_dataset(ruta)
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    return df_normalized, scaler

def agglomerative_partition_counts(n_clusters):
    df_normalized, _ = load_normalized_dataset()
    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(df_normalized)
    conteo = Counter(clusters)
    return conteo

# === EJECUCIÓN ===
if __name__ == "__main__":
    print("=" * 60)
    print("   CLUSTERING JERÁRQUICO AGLOMERATIVO SOBRE EL DRY BEAN DATASET")
    print("=" * 60)

    # Cargar datos originales
    df_original = load_clean_dataset()
    total_muestras = len(df_original)
    num_atributos = df_original.shape[1]

    print(f"\n📂 Dataset: Dry Bean Dataset (UCI Machine Learning Repository)")
    print(f"🔢 Total de muestras: {total_muestras}")
    print(f"📊 Total de atributos (sin la clase): {num_atributos}")
    print(f"📌 Algoritmo utilizado: Agglomerative Clustering (sin supervisión)")
    print(f"⚙️ Método de enlace predeterminado: 'ward' (scikit-learn)")
    print(f"\n🔍 Objetivo: Agrupar muestras de frijoles según sus características morfológicas.")
    
    # Ejecutar particiones
    for k in [6, 7, 8]:
        conteo = agglomerative_partition_counts(k)
        print(f"\n📁 Partición con {k} clusters:")
        print("-" * 40)
        for cluster_id, cantidad in sorted(conteo.items()):
            print(f"Cluster {cluster_id + 1}: {cantidad} elementos")
        print("-" * 40)

    print("\n✅ Proceso finalizado correctamente.")
