import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

from cleaning import load_clean_dataset, load_normalized_dataset

# Función para aplicar clustering aglomerativo y mostrar resultados
def clustering_aglomerativo(n_grupos):
    print("=" * 60)
    print(f"🔷 Clustering Jerárquico Aglomerativo - {n_grupos} grupos")
    print("=" * 60)

    # Cargar datos
    df_original = load_clean_dataset()
    df_normalized, _ = load_normalized_dataset()

    # Aplicar el modelo
    modelo = AgglomerativeClustering(n_clusters=n_grupos)
    etiquetas = modelo.fit_predict(df_normalized)

    # Añadir etiquetas al DataFrame original
    df_original["Grupo"] = etiquetas

    # Conteo de elementos por grupo
    conteo = Counter(etiquetas)
    total = len(df_original)

    print("\n📊 Distribución de muestras por grupo:")
    for grupo_id, cantidad in sorted(conteo.items()):
        porcentaje = (cantidad / total) * 100
        print(f"  ➤ Grupo {grupo_id + 1}: {cantidad} muestras ({porcentaje:.2f}%)")

    # Calcular Silhouette Score
    score = silhouette_score(df_normalized, etiquetas)
    print(f"\n⭐ Silhouette Score: {score:.4f}")

    # Visualización 2D con PCA (todos los datos)
    pca = PCA(n_components=2)
    reducidos = pca.fit_transform(df_normalized)

    print(f"\n🎯 Número de muestras visualizadas con PCA: {len(reducidos)}")

    # Gráfico PCA
    plt.figure(figsize=(10, 7))
    for grupo_id in sorted(set(etiquetas)):
        puntos = reducidos[etiquetas == grupo_id]
        plt.scatter(puntos[:, 0], puntos[:, 1],
                    label=f'Grupo {grupo_id + 1}',
                    s=12, alpha=0.6)
    plt.title(f'Visualización PCA - {n_grupos} Grupos Jerárquicos', fontsize=14)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title="Grupos", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico de barras
    plt.figure(figsize=(8, 5))
    grupos = [f'Grupo {gid + 1}' for gid in sorted(conteo)]
    cantidades = [conteo[gid] for gid in sorted(conteo)]
    plt.bar(grupos, cantidades, color='skyblue')
    plt.title(f'Cantidad de muestras por grupo - {n_grupos} grupos')
    plt.xlabel('Grupo')
    plt.ylabel('Número de muestras')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Exportar resultados
    nombre_archivo = f"resultados_jerarquico_{n_grupos}_grupos.csv"
    df_original.to_csv(nombre_archivo, index=False)
    print(f"\n📁 Resultados exportados: {nombre_archivo}")


# Ejecutar para 6, 7 y 8 grupos
if __name__ == "__main__":
    for n_grupos in [6, 7, 8]:
        clustering_aglomerativo(n_grupos)

    print("\n✅ Proceso completado.")
