
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    """Entraîne un modèle de classification simple"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Précision du modèle : {score:.2f}")
    return model

if __name__ == "__main__":
    print("Modèle prêt à être entraîné !")