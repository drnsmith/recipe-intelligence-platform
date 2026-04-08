
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import string

# Load dataset into a DataFrame
df = pd.read_csv('/path/to/recipes_data.csv')

# Initialise NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define cleaning utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define cleaning functions
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def convert_to_lowercase(text):
    return text.lower()

def remove_noncontext_words(text):
    text = text.replace('\n', ' ').replace('&nbsp', ' ')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text.strip()

def remove_short_words(text):
    return ' '.join([word for word in text.split() if len(word) > 3])

def remove_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_punctuation_and_newlines(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def lemmatize_text(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def stem_text(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(token) for token in tokens])

# Comprehensive preprocessing function
def preprocess_text(text):
    text = str(text)
    text = remove_non_ascii(text)
    text = convert_to_lowercase(text)
    text = remove_noncontext_words(text)
    text = remove_short_words(text)
    text = remove_tags(text)
    text = remove_numbers(text)
    text = remove_punctuation_and_newlines(text)
    text = lemmatize_text(text)
    text = stem_text(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(set(tokens))

# Apply preprocessing to relevant columns
df['preprocessed_ingredients'] = df['ingredients'].apply(preprocess_text)
df['preprocessed_directions'] = df['directions'].apply(preprocess_text)

# Combine columns for full recipe representation
df['preprocessed_full_recipe'] = df['preprocessed_ingredients'] + ' ' + df['preprocessed_directions']

# Display the cleaned data
print(df[['preprocessed_directions', 'preprocessed_ingredients', 'preprocessed_full_recipe']])
