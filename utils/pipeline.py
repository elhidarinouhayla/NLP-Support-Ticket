import pandas as pd
from utils.embeddings import load_embedding_model, generate_embedding, normalize_embeddings
from utils.training import prepare_data, train_model, metrics


def main():

    # load data
    df = pd.read_csv("./data/raw/dataset.csv")  

    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    texts = df["text"]       
    labels = df["type"]      

    # load embedding model
    model_emb = load_embedding_model()

    #  embeddings
    embeddings = generate_embedding(model_emb, texts)

    # normalize
    embeddings = normalize_embeddings(embeddings)

    # split data
    x_train, x_test, y_train, y_test = prepare_data(embeddings, labels)

    # train model
    model = train_model(x_train, y_train)

    # metrics
    accuracy, report = metrics(model, x_test, y_test)

    print(f"Accuracy: {accuracy}")
    print(report)

    print(" Pipeline finished successfully!")


if __name__ == "__main__":
    main()