from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb


def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model



def generate_embedding(model, texts):

    return model.encode(texts)




def normalize_embeddings(embeddings):

    embeddings_nor = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings_nor



def insexation_chroma(embeddings, df, collection_name="tickets_embeddings", batch_size=5000):

    client = chromadb.Client()
    collection = client.create_collection(collection_name)

    ids = [str(i) for i in df.index]
    embeddings_list = embeddings.tolist()
    metadatas = df[['type', 'queue', 'priority', 'language', 'tag_1', 'tag_2', 'tag_3', 'tag_4']].to_dict(orient='records')


    total = len(ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings_list[i:end],
            metadatas=metadatas[i:end]
        )
    return collection