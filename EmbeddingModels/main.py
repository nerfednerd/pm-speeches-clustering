from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

folder_path = "C:/Users/91982/OneDrive/Desktop/chain-models/EmbeddingModels/pmspeeches"

texts = []
file_names = []

for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read().strip()
            texts.append(text)
            file_names.append(file)

embeddings = [embedding.embed_query(t) for t in texts]

df = pd.DataFrame({"file": file_names, "text": texts, "embedding": embeddings})

df.to_csv("speeches_embeddings.csv", index=False)
print("Saved embeddings to speeches_embeddings.csv")




