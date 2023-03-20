import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Excel file
df = pd.read_excel(r'C:\\Users\Amorii7.AMORII7\Desktop\\vicinitas_search_results-2.xlsx')

# Get the text data as a list
documents = df['Name'].tolist()
#documents = df.iloc[:, 3].tolist()

# Replace NaN values with an empty string
df = df.fillna('')

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()

# Loop through each column and extract the text data
#for col in df.columns:
    # Get the text data as a list
    #documents = df[col].astype(str).tolist()

# Fit the vectorizer on the documents
vectorizer.fit(documents)

# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Loop through each document and print its TF-IDF features
for i, document in enumerate(documents):
    print(f"Document {i+1}:")
    tf_idf_vector = vectorizer.transform([document])
    for j, feature_index in enumerate(tf_idf_vector.indices):
        feature_name = feature_names[feature_index]
        feature_score = tf_idf_vector.data[j]
        print(f"  {feature_name}: {feature_score}")