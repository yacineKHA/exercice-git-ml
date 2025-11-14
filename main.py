
from data.load_data import load_dataset
from models.train_model import train_model

def main():
    print("=== Pipeline Machine Learning ===")

    # Chargement des données
    df = load_dataset()
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # Entraînement du modèle
    model = train_model(X, y)
    
    print("Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()