from data_preprocessing import preprocess_data
from gnn_model import train_gnn
from evaluation import evaluate_model

def main():
    # Preprocess data
    G, features, labels = preprocess_data()

    # Train GNN model
    model, embedding = train_gnn(G, features, labels)

    # Evaluate the model
    evaluate_model(model, embedding, labels)

if __name__ == "__main__":
    main()
