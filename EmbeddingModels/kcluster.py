import pandas as pd
import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load CSV
df = pd.read_csv("speeches_embeddings.csv")

# Convert the "embedding" string back into a list of floats
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Extract embeddings as a list of lists
embeddings = df["embedding"].tolist()

# Cluster into 2 groups (change n_clusters as needed)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels to DataFrame
df["cluster"] = clusters

# Show which file/speech belongs to which cluster
print(df[["file", "cluster"]])   # <-- file column from your earlier script

# Reduce embeddings to 2D with PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(reduced[:,0], reduced[:,1], c=df["cluster"], cmap="viridis")

# Add labels (use filename instead of full text)
for i, fname in enumerate(df["file"]):
    plt.annotate(fname, (reduced[i,0], reduced[i,1]))

plt.title("PM Independence Day Speeches Clusters (PCA)")
plt.show()
