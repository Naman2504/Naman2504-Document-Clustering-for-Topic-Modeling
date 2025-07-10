# Naman2504-Document-Clustering-for-Topic-Modeling
📂 Document Clustering and Topic Modeling with LDA & KMeans
This project applies Latent Dirichlet Allocation (LDA) for topic modeling and KMeans for document clustering on the 20 Newsgroups dataset. It performs text preprocessing, TF-IDF vectorization, topic inference using both Scikit-learn and Gensim, visualizes topics using pyLDAvis, and clusters documents using KMeans and t-SNE.

📌 Features
Extracts and reads text data from a .tar.gz archive of newsgroup posts

Preprocesses documents (lowercasing, stopword removal, lemmatization)

Converts text to TF-IDF features

Applies:

LDA (Scikit-learn) for topic modeling

LDA (Gensim) for top topic documents & visualization

KMeans for document clustering

Visualizes topic models (pyLDAvis) and document clusters (matplotlib)

Saves outputs as CSV and HTML for analysis

📁 Folder Structure
bash
Copy
Edit
.
├── document_clustering.py      # Main script
├── 20_newsgroups.tar.gz        # Dataset archive (manually downloaded)
├── lda_topics_gensim.html      # Interactive LDA visualization
├── top_documents_per_topic.csv # Top docs per topic from Gensim
├── document_clusters.csv       # KMeans clustering results
├── kmeans_clusters.png         # Cluster visualization (2D)
└── README.md                   # This file
🛠️ Requirements
Python 3.7+

Libraries:

bash
Copy
Edit
pip install numpy pandas matplotlib nltk gensim scikit-learn pyLDAvis
First-time NLTK setup:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
📦 Dataset
Download the 20 Newsgroups dataset and place the archive (20_newsgroups.tar.gz) in the project directory.

This script works on the misc.forsale category. You can change the category by editing this line:

python
Copy
Edit
category_path = os.path.join(extracted_path, 'misc.forsale')
🚀 How to Run
bash
Copy
Edit
python document_clustering.py
Outputs:

📄 document_clusters.csv – Cluster ID for each document

📄 top_documents_per_topic.csv – Top 5 documents per topic

📊 kmeans_clusters.png – 2D t-SNE visualization of clusters

🌐 lda_topics_gensim.html – Interactive LDA topic explorer

📈 Visualizations
pyLDAvis for exploring LDA topics:

Opens in browser as an interactive HTML

t-SNE Plot for KMeans clusters:

Displays document clusters in 2D space

📊 Sample Output (CSV)
document_clusters.csv

Document	Cluster
...	3

top_documents_per_topic.csv

Topic	Document_ID	Topic_Probability	Text
0	12	0.91	"for sale..."

✅ To-Do / Improvements
Extend to all categories in the dataset

Add GUI for interactive document exploration

Support other vectorizers (e.g., CountVectorizer)

Optimize preprocessing with spaCy

📄 License
This project is licensed under the MIT License.
