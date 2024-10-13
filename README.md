This project implements a Graph Neural Network (GNN) model that uses random walks and Word2Vec to embed graph nodes. The model is trained and evaluated on a sample graph.

## File Structure
- `main.py`: Entry point for the project.
- `data_preprocessing.py`: Contains data preprocessing steps.
- `gnn_model.py`: Contains the GNN model and training functions.
- `evaluation.py`: Evaluation metrics for the trained GNN model.

## How to Run
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

## Dependencies
- NetworkX
- NumPy
- Pandas
- Gensim
- Scikit-learn
