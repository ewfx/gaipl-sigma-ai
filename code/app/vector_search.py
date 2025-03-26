from sentence_transformers import SentenceTransformer
import faiss

def build_vector_index(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings, model

def retrieve_similar_incidents(query_text, model, index, df, top_k=3):
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]].copy()
    return results