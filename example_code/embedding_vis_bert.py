from transformers import BertTokenizer, BertModel
import plotly.express as px
from sklearn.decomposition import PCA

# 1. load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 2. select words and get corresponding embeddings (using [CLS] token vector)
words = words = [
    "king", "queen", "man", "woman", "boy", "girl",
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "animal",
    "car", "truck", "bus", "vehicle",
    "happy", "sad", "joy", "anger"
]
embeddings = []

for word in words:
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    embeddings.append(vec)

# 3. reduce dimensions to 3D using PCA
from sklearn.decomposition import PCA
X_3d = PCA(n_components=3).fit_transform(embeddings)

# 4. visualize using plotly
fig = px.scatter_3d(
    x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
    text=words, color=words, title="BERT Word Embedding (3D PCA)"
)
fig.show()
