import pandas as pd

def load_dataset():
    """Charge le dataset pour l'entraînement"""
    # Exemple de données fictives
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    print(f"Dataset chargé : {len(df)} lignes")
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())