from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings using BERT
def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings to represent the sentence
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

# Generate embeddings for the preprocessed recipes
df['bert_embeddings'] = df['preprocessed_full_recipe'].apply(generate_bert_embeddings)

# Compute similarity matrix
similarity_matrix = cosine_similarity(list(df['bert_embeddings']))

# Optional: Apply dimensionality reduction for clustering
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(list(df['bert_embeddings']))

# Save reduced embeddings back to the DataFrame
df['reduced_embeddings'] = list(reduced_embeddings)

# Display embeddings and similarity results
print("Similarity Matrix:")
print(similarity_matrix)
print("Reduced Embeddings:")
print(reduced_embeddings[:5])
