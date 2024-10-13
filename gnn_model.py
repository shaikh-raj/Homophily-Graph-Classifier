from gensim.models import Word2Vec
import random
import networkx as nx

def train_gnn(G, features, labels):
    # Step 2: Generate random walks
    walks = generate_walks(G, num_walks=10, walk_length=80)

    # Train Word2Vec on walks
    model = Word2Vec(walks, vector_size=64, window=5, min_count=0, sg=1, workers=4)
    
    # Placeholder for GNN training logic
    embedding = model.wv  # Get node embeddings
    
    return model, embedding

def generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(node))
                if neighbors:
                    node = random.choice(neighbors)
                    walk.append(node)
            walks.append(walk)
    return walks
