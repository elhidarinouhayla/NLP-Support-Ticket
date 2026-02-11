from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb


def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model



def generate_embedding(model, texts):

    return model.encode(texts)




def normlize_embeddings(embeddings):

    embeddings_nor = embeddings / np.linalg.norm(embeddings)

    return embeddings_nor


def save_chroma(embeddings, df, collection_name="tickets_embeddings"):

    client = chromadb.Client()
    collection = client.create_collection(collection_name)

    collection.add(
        ids=list(df.index),
        embeddings=embeddings,
        metadatas=df[['type', 'queue', 'priority']].to_dict(orient='records')
    )


    return collection