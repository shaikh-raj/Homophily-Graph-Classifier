import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def evaluate_model(model, embedding, labels):
    # Use node embeddings and calculate metrics
    node_embeddings_df = pd.DataFrame(embedding)
    node_embeddings_df['total'] = node_embeddings_df.sum(axis=1)
    
    # Assign class based on sum of embeddings
    node_embeddings_df['class'] = node_embeddings_df['total'].apply(lambda x: 1 if x > 0 else 0)
    
    # Compute evaluation metrics
    acc = accuracy_score(labels, node_embeddings_df['class'])
    print(f"Accuracy: {acc}")
    
    # Additional metrics
    precision = precision_score(labels, node_embeddings_df['class'])
    recall = recall_score(labels, node_embeddings_df['class'])
    cm = confusion_matrix(labels, node_embeddings_df['class'])
    
    print(f"Precision: {precision}, Recall: {recall}")
    print(f"Confusion Matrix:\n {cm}")
