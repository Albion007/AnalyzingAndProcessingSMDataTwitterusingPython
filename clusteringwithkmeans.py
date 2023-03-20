import numpy as np
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from wordcloud import WordCloud
import sklearn
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

f = open(r"C:\\Users\Amorii7.AMORII7\Desktop\\vicinitas_search_results-2.xlsx", "r",encoding='utf-8', errors='ignore')
text1 = f.read()
print(text1)

stop_words = set(stopwords.words('english'))
string = ''.join(str(x) for x in text1)
documents = word_tokenize(string)
#print(documents)
wordcloud=WordCloud(stopwords=stop_words,
                    background_color='white').generate(text1)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
#plt.show()

vec=TfidfVectorizer(stop_words='english',use_idf=True)
vec.fit(documents)
features=vec.transform(documents)

#elbow method
true_k=4
cls=MiniBatchKMeans(n_clusters=true_k,
                    random_state=True).fit(features)
cls.labels_
cls.predict(features)

kmeans_kwargs={"init":"random", "n_init":10,
               "max_iter":300,"random_state":42}
#sum squared error
sse=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)
plt.plot(range(1,11),sse)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.savefig('elbow.png')

sorted_centroids=cls.cluster_center_.argsort()[:, ::-1]
terms=vec.get_feature_names_out()
for i in range(true_k):
    print("Cluster %d: "% i, end='')
    for ind in sorted_centroids[i, 5]:
        print('%s'% terms[ind], end='')
    print()
    print()
print()
