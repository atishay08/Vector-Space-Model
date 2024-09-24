import os
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only need to do this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to load documents from the corpus folder
def load_corpus(corpus_folder):
    corpus = []
    doc_ids = []
    for filename in os.listdir(corpus_folder):
        filepath = os.path.join(corpus_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read().lower().split()
            corpus.append(content)
            doc_ids.append(filename)
    return corpus, doc_ids

# Function to compute term frequency (tf)
def compute_tf(doc):
    tf = defaultdict(int)
    for term in doc:
        tf[term] += 1
    return {term: 1 + math.log10(freq) for term, freq in tf.items()}

# Function to compute inverse document frequency (idf)
def compute_idf(corpus):
    N = len(corpus)
    df = defaultdict(int)
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    return {term: math.log10(N / df[term]) for term in df}

# Function to compute tf-idf for a document
def compute_tf_idf(tf, idf):
    return {term: tf[term] * idf.get(term, 0) for term in tf}

# Function to normalize a vector (used in cosine similarity)
def normalize(vector):
    norm = math.sqrt(sum(weight ** 2 for weight in vector.values()))
    return {term: weight / norm for term, weight in vector.items()}

# Function to compute cosine similarity between query and document
def cosine_similarity(query_vec, doc_vec):
    intersection = set(query_vec.keys()) & set(doc_vec.keys())
    return sum(query_vec[term] * doc_vec[term] for term in intersection)

# Function to rank documents based on cosine similarity
def rank_documents(query, corpus, tf_idf_corpus, idf, doc_ids):
    tf_query = compute_tf(query)
    tf_idf_query = compute_tf_idf(tf_query, idf)
    normalized_query = normalize(tf_idf_query)

    similarities = []
    for doc_id, doc_tf_idf in enumerate(tf_idf_corpus):
        normalized_doc = normalize(doc_tf_idf)
        similarity = cosine_similarity(normalized_query, normalized_doc)
        similarities.append((doc_ids[doc_id], similarity))
    
    # Sort by relevance and return top 10 results
    return sorted(similarities, key=lambda x: (-x[1], x[0]))[:10]

# Function to preprocess the query
def preprocess_query(query):
    tokens = word_tokenize(query.lower())  # Tokenizing the query
    tokens = [word for word in tokens if word.isalnum()]  # Removing punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Removing stop words
    return [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization

# Updated path to the corpus folder
corpus_folder = r'C:\Users\atish\Desktop\IR Assignments\Assignment2\Corpus'

# Load the corpus and document IDs
corpus, doc_ids = load_corpus(corpus_folder)

# Print loaded corpus and document IDs for verification
print("Loaded Corpus:", corpus)
print("Document IDs:", doc_ids)

# Preprocess queries
query_1_text = "Developing your Zomato business account and profile is a great way to boost your restaurant’s online reputation"
query_1 = preprocess_query(query_1_text)

query_2_text = "Warwickshire, came from an ancient family and was the heiress to some land"
query_2 = preprocess_query(query_2_text)

# Compute IDF for the entire corpus
idf = compute_idf(corpus)
print("IDF values:", idf)

# Compute TF-IDF for each document in the corpus
tf_idf_corpus = [compute_tf_idf(compute_tf(doc), idf) for doc in corpus]
print("TF-IDF for first document:", tf_idf_corpus[0])

# Rank documents for Query 1
print("Top 10 documents for Query 1:")
results_1 = rank_documents(query_1, corpus, tf_idf_corpus, idf, doc_ids)
for rank, (doc_id, score) in enumerate(results_1, start=1):
    print(f"{rank}. {doc_id}: {score:.4f}")

# Rank documents for Query 2
print("\nTop 10 documents for Query 2:")
results_2 = rank_documents(query_2, corpus, tf_idf_corpus, idf, doc_ids)
for rank, (doc_id, score) in enumerate(results_2, start=1):
    print(f"{rank}. {doc_id}: {score:.4f}")
