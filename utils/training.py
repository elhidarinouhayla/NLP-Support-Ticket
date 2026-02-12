import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def prepare_data(embeddings, labels):

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)


    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    
    return model


def metrics(model, x_test, y_test):

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report