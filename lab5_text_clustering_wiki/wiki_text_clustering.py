from collections import defaultdict

import matplotlib.pyplot as plt
import spacy
import wikipedia
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import save_npz
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# 1. Сбор статей
TOPICS = ["Classical music", "Film", "Painting"]
NUM_PER_TOPIC = 15

wikipedia.set_lang("en")
article_texts, article_titles, article_topics = [], [], []

for topic in TOPICS:
    results = wikipedia.search(topic, results=50)
    cnt = 0
    for title in results:
        if cnt >= NUM_PER_TOPIC:
            break
        try:
            page = wikipedia.page(title)
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue
        text = page.content
        if not text:
            continue
        article_texts.append(text)
        article_titles.append(title)
        article_topics.append(topic)
        cnt += 1
    print(f"Loaded {cnt} articles for topic «{topic}»")

# 2. Предобработка текста
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
clean_texts = []
for raw in article_texts:
    doc = nlp(raw.lower())
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
    clean_texts.append(" ".join(tokens))

# 3. TF-IDF векторизация
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(clean_texts)
print("TF-IDF matrix shape:", X.shape)

print("TF-IDF matrix (dense):")
print(X.toarray())

save_npz("tfidf_matrix.npz", X)

# 4. Метод локтя
inertias = []
ks = list(range(1, 11))
for k in ks:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42)
    km.fit(X.toarray())
    inertias.append(km.inertia_)
plt.figure(figsize=(6, 4))
plt.plot(ks, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.close()
print("Saved elbow_plot.png")

k_opt = 3  # локоть показывает 3

# 5. K-Means + PCA
labels_km = KMeans(n_clusters=k_opt, random_state=42).fit_predict(X.toarray())
X_pca = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap="tab10", s=50)
plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("kmeans_pca.png")
plt.close()
print("Saved kmeans_pca.png")

# 6. Иерархическая кластеризация
Z = linkage(X.toarray(), method="ward")
plt.figure(figsize=(8, 6))
dendrogram(
    Z, labels=[t[:20] for t in article_titles], leaf_rotation=90, leaf_font_size=8
)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.tight_layout()
plt.savefig("dendrogram.png")
plt.close()
print("Saved dendrogram.png")

# 7. TSNE + DBSCAN
tsne_emb = TSNE(n_components=2, random_state=42).fit_transform(X.toarray())
labels_db = DBSCAN(eps=0.65, min_samples=3).fit_predict(tsne_emb)
plt.figure(figsize=(6, 6))
plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels_db, cmap="tab10", s=50)
plt.title("DBSCAN on t-SNE Embedding")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("dbscan_tsne.png")
plt.close()
print("Saved dbscan_tsne.png")


# 8. Вывод и сохранение результатов
def format_clusters(labels, titles):
    clusters = defaultdict(list)
    for lbl, ttl in zip(labels, titles):
        clusters[lbl].append(ttl)
    out = []
    for cid, docs in sorted(clusters.items()):
        out.append(f"Cluster {cid}:")
        out += [f"  - {d}" for d in docs]
    return "\n".join(out)


labels_hc = AgglomerativeClustering(n_clusters=k_opt).fit_predict(X.toarray())

print("\n--- K-Means Clustering ---")
print(format_clusters(labels_km, article_titles))
print("\n--- Hierarchical Clustering ---")
print(format_clusters(labels_hc, article_titles))
print("\n--- DBSCAN Clustering (t-SNE) ---")
print(format_clusters(labels_db, article_titles))

with open("clustering_results.txt", "w", encoding="utf-8") as f:
    f.write("--- K-Means Clustering ---\n")
    f.write(format_clusters(labels_km, article_titles) + "\n\n")
    f.write("--- Hierarchical Clustering ---\n")
    f.write(format_clusters(labels_hc, article_titles) + "\n\n")
    f.write("--- DBSCAN Clustering (t-SNE) ---\n")
    f.write(format_clusters(labels_db, article_titles) + "\n")

# 9. Сохранение текстов
with open("articles_raw.txt", "w", encoding="utf-8") as f_raw, open(
    "articles_cleaned.txt", "w", encoding="utf-8"
) as f_clean:
    last = None
    for tp, ttl, raw, clean in zip(
        article_topics, article_titles, article_texts, clean_texts
    ):
        if tp != last:
            f_raw.write(f"\n>>> Topic: {tp} <<<\n")
            f_clean.write(f"\n>>> Topic: {tp} <<<\n")
            last = tp
        f_raw.write(f"\n# {ttl}\n{raw}\n")
        f_clean.write(f"\n# {ttl}\n{clean}\n")

print("All results, texts and clusters saved.")
