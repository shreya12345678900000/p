from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report 
import numpy as np 
# ---------- PART 1 ---------- 
docs = [ 
"The cat sat on the mat.", 
"The dog sat on the mat.", 
"The cat chased the mouse.", 
"Dogs and cats are great pets.", 
"I love my pet dog." 
] 
vec = TfidfVectorizer(stop_words='english') 
tfidf = vec.fit_transform(docs) 
print("\nTF-IDF Matrix:\n", tfidf.toarray()) 
sim = cosine_similarity(tfidf) 
print("\nCosine Similarity:\n", sim) 
most_similar = np.argsort(sim[0])[::-1][1] 
print("\nMost similar to Doc1 → Doc", most_similar + 1) 
# ---------- PART 2 ---------- 
texts = [ 
"I love playing football","The game of soccer is amazing","Soccer is a popular sport", 
"Football players train hard","The team won the championship","He kicked the ball into the goal", 
"Fans cheered during the soccer match","The referee blew the whistle", 
"Training improves football performance","They scored three goals yesterday", 
"Python and machine learning are fun","Deep learning improves neural networks", 
"Artificial intelligence and data science","Neural networks can recognize images", 
"I enjoy programming in Python","Data analysis is key in AI research", 
"Machine learning models need lots of data","AI helps in automating tasks", 
"Natural language processing is a part of AI","Big data drives modern technology" 
] 
labels = ["sports"]*10 + ["tech"]*10 
v = TfidfVectorizer(stop_words="english") 
X = v.fit_transform(texts) 
X_train, X_test, y_train, y_test = train_test_split( 
X, labels, test_size=0.3, random_state=42, stratify=labels 
) 
model = MultinomialNB().fit(X_train, y_train) 
pred = model.predict(X_test) 
print("\nClassification Report:\n", classification_report(y_test, pred, zero_division=0)) 
