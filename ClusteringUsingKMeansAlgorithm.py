import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the Excel file
df = pd.read_excel(r'C:\\Users\Amorii7.AMORII7\Desktop\\vicinitas_search_results-2.xlsx')

# Replace NaN values with an empty string
df = df.fillna('')

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Get the text data as a list
documents = df.stack().astype(str).tolist()

# Fit the vectorizer on the documents
vectorizer.fit(documents)

# Transform the documents into a TF-IDF matrix
tf_idf_matrix = vectorizer.transform(documents)

# Perform clustering using K-Means algorithm
kmeans = KMeans(n_clusters=10, random_state=42).fit(tf_idf_matrix)

# Print the cluster labels for each document
for i, document in enumerate(documents):
    cluster_label = kmeans.labels_[i]
    print(f"Document {i+1} (in column {df.columns[i%len(df.columns)]}) belongs to cluster {cluster_label}")