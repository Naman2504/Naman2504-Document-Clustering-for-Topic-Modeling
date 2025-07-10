import os
import tarfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import re
import matplotlib.pyplot as plt
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

# Download NLTK data files (only for the first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data_path = r"C:\Users\kk\Desktop\document clustering\20_newsgroups.tar.gz"
extracted_path = r"C:\Users\kk\Desktop\document clustering\20_newsgroups"

if not os.path.exists(extracted_path):
    with tarfile.open(data_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(data_path))

# Load the data from the specific category
category_path = os.path.join(extracted_path, 'misc.forsale')
documents = []

for file_name in os.listdir(category_path):
    file_path = os.path.join(category_path, file_name)
    with open(file_path, 'r', encoding='latin1') as file:
        documents.append(file.read())

# Preprocess the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

documents = [preprocess(doc) for doc in documents]

# Convert the text data into numerical form using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import pyLDAvis

# Tokenize preprocessed documents
tokenized_docs = [doc.split() for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(tokenized_docs)
gensim_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Train Gensim LDA model
gensim_lda = LdaModel(corpus=gensim_corpus, id2word=dictionary, num_topics=10, passes=10, random_state=42)

# Number of top documents per topic to save
TOP_N = 5

# Get the dominant topic for each document
doc_topics = []
for doc_bow in gensim_corpus:
    topics = gensim_lda.get_document_topics(doc_bow)
    dominant_topic = max(topics, key=lambda x: x[1])[0]
    doc_topics.append(dominant_topic)

# Create DataFrame with documents and their dominant topic
df_docs = pd.DataFrame({
    'Document': documents,
    'Dominant_Topic': doc_topics
})

# Get top documents for each topic by topic probability
top_docs_per_topic = []

for topic_id in range(gensim_lda.num_topics):
    topic_docs = []
    for i, doc_bow in enumerate(gensim_corpus):
        topic_probs = gensim_lda.get_document_topics(doc_bow)
        for t_id, prob in topic_probs:
            if t_id == topic_id:
                topic_docs.append((i, prob))
                break
    # Sort by topic probability and get top N
    top_docs = sorted(topic_docs, key=lambda x: -x[1])[:TOP_N]
    for doc_id, prob in top_docs:
        top_docs_per_topic.append({
            'Topic': topic_id,
            'Document_ID': doc_id,
            'Topic_Probability': prob,
            'Text': documents[doc_id]
        })

# Convert to DataFrame
df_top_docs = pd.DataFrame(top_docs_per_topic)

# Save to CSV
df_top_docs.to_csv('top_documents_per_topic.csv', index=False)
print("✅ Top documents per topic saved to top_documents_per_topic.csv")

# Prepare and save visualization
vis_data = pyLDAvis.gensim_models.prepare(gensim_lda, gensim_corpus, dictionary)
pyLDAvis.save_html(vis_data, 'lda_topics_gensim.html')

print("✅ LDA visualization saved to lda_topics_gensim.html")


# Display the topics found by LDA
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# Apply K-means for document clustering
num_clusters = 10
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(X)

# Reduce TF-IDF features to 2D using t-SNE
tsne_model = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne_model.fit_transform(X.toarray())

# Plot clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=km.labels_, cmap='tab10', alpha=0.7)
plt.title("KMeans Clustering of Documents (t-SNE Projection)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.show()

# Attach the cluster labels to the documents
df = pd.DataFrame({'Document': documents, 'Cluster': km.labels_})

# Display the clustering results
print(df.head())

# Save the clustering results to a CSV file
df.to_csv('document_clusters.csv', index=False)