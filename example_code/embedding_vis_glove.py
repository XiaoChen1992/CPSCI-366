import gensim.downloader as api
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# 1. load GloVe model
model = api.load("glove-wiki-gigaword-50")

# 2. select words to visualize
words = [
    "king", "queen", "man", "woman", "boy", "girl",
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "animal",
    "car", "truck", "bus", "vehicle",
    "happy", "sad", "joy", "anger"
]

# 3. get embeddings for the selected words
X = np.array([model[word] for word in words])

# 4. reduce dimensions to 3D using t-SNE or PCA
use_tsne = True  # Set to False to use PCA instead
if use_tsne:
    reducer = TSNE(n_components=3, random_state=42, perplexity=10)
else:
    reducer = PCA(n_components=3)
X_embedded = reducer.fit_transform(X)

# 5. use plotly to create a 3D scatter plot
fig = px.scatter_3d(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    z=X_embedded[:, 2],
    text=words,
    color=words,  
    title="3D Word Embedding Visualization (t-SNE)"
)
fig.update_traces(marker=dict(size=6))
fig.show()
