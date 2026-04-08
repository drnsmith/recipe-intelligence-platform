
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings
def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

# Apply embedding generation
def create_embeddings(file_path):
    df = pd.read_csv(file_path)
    df['bert_embeddings'] = df['processed_full_recipe'].apply(generate_bert_embeddings)
    df.to_pickle('bert_embeddings.pkl')
    print("BERT embeddings saved to 'bert_embeddings.pkl'.")

if __name__ == "__main__":
    file_path = 'processed_recipes.csv'
    create_embeddings(file_path)
