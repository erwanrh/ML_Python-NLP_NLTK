#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing task
# Using the NLTK package

# In[1]:


import nltk


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('movie_reviews')


# In[2]:


from nltk.corpus import movie_reviews

categories = movie_reviews.categories()

sentences = []
targets = []

for file_id in movie_reviews.fileids():
    words = movie_reviews.words(file_id)
    sentences.append(' '.join(words))
    targets.append(categories.index(movie_reviews.categories(file_id)[0]))


# # Part 1- Tokenize

# In[3]:


text_tokenized = [nltk.word_tokenize(text) for text in sentences]


# In[4]:


print(text_tokenized[0])


# In[5]:


len(text_tokenized)


# ## Part 2 - Remove punctuation and special characters

# In[51]:


import string

punct = [a for a in string.punctuation]

text_no_punctuation = []


# In[52]:


for sentence in text_tokenized:
    sentence_new = [word for word in sentence if word not in punct]
    text_no_punctuation.append(sentence_new)


# In[53]:


len(text_no_punctuation)


# ## Part 3 - Remove stop words

# In[54]:


from nltk.corpus import stopwords

text_no_stop_words = []


# In[55]:


stoplist = stopwords.words('english')

for sentence in text_no_punctuation:
    text_no_stop_words.append([word for word in sentence if word not in stoplist])


# In[56]:


print('Number of words before removing stop words: %d' %len(text_no_punctuation[0]))
print('Number of words after removing stop words: %d' %len(text_no_stop_words[0]))

print(len(text_no_stop_words[0]))


# ## Part 4 - Stemming

# In[57]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
text_stemmed = []


# In[58]:


for sentence in text_no_stop_words:
    text_stemmed.append([porter.stem(word) for word in sentence])


# In[59]:


text_stemmed[0]


# ## Part 5 - Bag of words

# In[60]:


from sklearn.feature_extraction.text import CountVectorizer


# In[61]:


clean_sentences = [' '.join(sentence) for sentence in text_stemmed]


# In[65]:


from sklearn.model_selection import train_test_split

sentence_train, sentence_test, y_train, y_test = train_test_split(clean_sentences, targets, shuffle=True, test_size=0.1)


# In[67]:


count_vectorizer = CountVectorizer()
X_train = count_vectorizer.fit_transform(sentence_train)
X_test = count_vectorizer.transform(sentence_test)


# ## Part 6 - Random Forest Classifier

# In[68]:


from sklearn.ensemble import RandomForestClassifier


# In[69]:


rm_model = RandomForestClassifier()

rm_model.fit(X_train, y_train)


# In[71]:


preds_train = rm_model.predict(X_train)
print('Accuracy for training set is : %.4f' % (sum(y_train == preds_train)/len(preds_train)))

preds_test = rm_model.predict(X_test)
print('Accuracy for test set is : %.4f' % (sum(y_test == preds_test)/len(preds_test)))


# ## Part 7 - glove (vector representation of words)

# In[13]:


import gensim


# In[25]:


from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("u011798\Downloads\23\model.bin", binary=True)


# In[26]:


model.most_similar('mathematician')


# In[27]:


model.most_similar('sad')


# ## Part 8 - Linear algebra with words

# In[28]:


model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])


# In[29]:


model.most_similar_cosmul(positive=['happy', 'negative'], negative=['positive'])


# ## Part 9 - Visualize

# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn.manifold import TSNE

def plot_words(word1, word2, model, n=100):    
    words_1 = model.most_similar(word1, topn=n)
    words_2 = model.most_similar(word2, topn=n)
    words_1 = [similar[0] for similar in words_1]
    words_1.append(word1)
    words_2 = [similar[0] for similar in words_2]
    words_2.append(word2)
    words1 = [model.get_vector(similar) for similar in words_1]
    words2 = [model.get_vector(similar) for similar in words_2]

    vectors = words1.copy()
    vectors.extend(words2)

    words = words_1.copy()
    words.extend(words_2)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(vectors)
    tsne_df = pd.DataFrame({'X': tsne_result[:,0], 'Y': tsne_result[:,1], 'word_similarity': [word1]*(n+1)+[word2]*(n+1)})

    sns.lmplot('X', 'Y', tsne_df, hue='word_similarity', fit_reg=False)
    plt.annotate(word1, (tsne_result[words.index(word1), 0], tsne_result[words.index(word1), 1]))
    plt.annotate(word2, (tsne_result[words.index(word2), 0], tsne_result[words.index(word2), 1]))

    plt.show()


# In[31]:


plot_words('computer', 'doctor', model)


# In[32]:


plot_words('history', 'mathematics', model)


# In[ ]:




