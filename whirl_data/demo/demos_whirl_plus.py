"""
Whirl ProbLog implementation in animal dataset

Builds a LDA model in the background using gensim, NLTK and Scikit-learn, such
that ProbLog can make use of document similarity while reasoning.

From:
W. W. Cohen. Whirl: A word-based information representation language.
Artificial Intelligence, 118(1):163-196, 2000.

Author:
- Wannes Meert
- Anton Dries
Modified by Hikaru
TF-IDF -> LDA
"""
from __future__ import print_function

import os, sys
import string
import glob
import time
import re
import codecs

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from stop_words import get_stop_words
from gensim import corpora, models, similarities, matutils

#sys.path.append('../../')
#sys.path.append(os.path.abspath('F:\\Program Files\\Anaconda\\Lib\\site-packages\\'))

from problog.extern import problog_export_nondet


# Settings
#texts = glob.glob('E:\\GitHub\\document-soft-join-in-problog\\whirl_data\\demo\\*.txt') 
filePath = "E:\\GitHub\\document-soft-join-in-problog\\whirl_data\\demo\\"
baseFile = "demos_base.dic"
texts = filePath + "demos.txt"  # input texts for TF-IDF model

# Global variables
dictionary = None # Dictionary for lda 
corpus = None # Corpus for lda
lda = None # LDA model
tfidf = None # tfidf model
# tokens = None # List of tokens used by LDA
similarity_index = None # index of all documents to be compared 

texts_dic = {} # Dictionary to store all texts
# texts_key = {} # Dictionary to store all texts_key pairs of base case

data_frame = pd.read_table(filePath + baseFile, compression = None, header = None, names = ['relation', 'key','name'])


if sys.version_info.major == 2:
  punct_table = string.maketrans('','')
else:
  punct_table = str.maketrans('','', string.punctuation)
stemmer = PorterStemmer()

def cleantext(text):
  """Clean string from punctuation and capitals."""
  lowers = text.lower()
  if sys.version_info.major == 2:
    no_punctuation = lowers.translate(punct_table, string.punctuation)
  else:
    no_punctuation = lowers.translate(punct_table)
  return no_punctuation


def tokenize(text):
  """Transform string to list of stemmed tokens."""
  # tokenize cleanned document string
  tokens = nltk.word_tokenize(text)
  # remove stop words from tokens
  en_stop = get_stop_words('en')
  stopped_tokens = [i for i in tokens if not i in en_stop]
  # stem tokens
  stemmed = [stemmer.stem(token) for token in stopped_tokens]
  return stemmed


def getLDA():
  """Return cached LDA model."""
  global dictionary
  global lda
  global tokens
  global corpus
  global similarity_index
  global tfidf

  if lda is None:

    #turn tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary.load(filePath + 'lda_dictionary.dict')
    corpus = corpora.MmCorpus(filePath + 'lda_corpus.mm')
    # generate LDA model, num_topics need to be considered
    lda = models.ldamodel.LdaModel(corpus, num_topics = 15, id2word = dictionary, passes = 20)
    similarity_index = similarities.MatrixSimilarity.load(filePath + 'lda_similarity.index')
    tfidf = models.tfidfmodel.TfidfModel(corpus, normalize = True)
    
  return lda

def initialize():
  """Create all necessary dictionary, corpus, similarity index"""
  global dictionary
  global lda
  global corpus
  global similarity_index

  texts_content = []
  with open(texts, 'r') as ifile:
    for line in ifile:
      line1 = line[:-1]
      texts_content.append(cleantext(line1))
  # list for tokenized documents
  tokens = []
  for text in texts_content:
    tokens.append(tokenize(text))
  #turn tokenized documents into a id <-> term dictionary
  dictionary = corpora.Dictionary(tokens)
  dictionary.save(filePath + 'lda_dictionary.dict')

  corpus = [dictionary.doc2bow(token) for token in tokens]
  corpora.MmCorpus.serialize(filePath + 'lda_corpus.mm', corpus)
  # generate LDA model, num_topics need to be considered
  lda = models.ldamodel.LdaModel(corpus, num_topics = 15, id2word = dictionary, passes = 20)

  # generate similarity index
  similarity_index = similarities.MatrixSimilarity(lda[corpus])
  similarity_index.save(filePath + 'lda_similarity.index')

def bestTfidfMatch(e, similar_list):
  """Use TF-IDF to find the similarity"""
  global texts_dic
  high_sim_score = similar_list[0][1]
  highest_cos_sim = 0
  best_match = ("None", "None", 0)
  text_from_base = None
   
  e_vec = getTfidfVec(e)

  for i in similar_list:
    if i[1] == high_sim_score:
      text_from_base = np.asarray(data_frame.iloc[i[0]])
      text_base_vec = getTfidfVec(text_from_base[2])
      
      cos_sim = matutils.cossim(text_base_vec, e_vec)
      if cos_sim > highest_cos_sim:
        highest_cos_sim = cos_sim
        best_match = (text_from_base[1], text_from_base[2], cos_sim)  #key, name, score
    else:
      break
  return best_match

def bestMatch(similar_list):
  """ return best similar result"""
  high_sim_score = similar_list[0][1]
  best_match = []
  for i in similar_list:
    if i[1] == high_sim_score:
      text_from_base = np.asarray(data_frame.iloc[i[0]])
      best_match.append((text_from_base[1], text_from_base[2], float('%.4f'% high_sim_score)))
  return best_match

def getTfidfVec(text):  
  global texts_dic
  global tfidf

  if not tfidf:
    tfidf = models.tfidfmodel.TfidfModel(corpus, normalize = True)
  if text in texts_dic.keys():
    text_vec = texts_dic[text]
  else:
    v = dictionary.doc2bow(tokenize(cleantext(text)))
    text_vec = tfidf[v]
    texts_dic[text] = text_vec
  
  return text_vec


@problog_export_nondet('+str', '-list')  #for similarity(Str1, Str2, P)
def similarity(e2):
  """TF-IDF similarity between two documents based on pre-processed texts.
     Expects two text/document encodings.
     Input: one query string.
     Output: identical string and similarity
  """
  global texts_dic
  
  lda = getLDA()
  output = ("None", "None", 0)
  v2_lda = lda[dictionary.doc2bow(tokenize(cleantext(e2)))]
  sims = similarity_index[v2_lda]
  sims = sorted(enumerate(sims), key = lambda item: -item[1])
  if sims[0][1] > 0.1:
    #output = bestTfidfMatch(e2, sims)
    output = bestMatch(sims)
  #return [output]
  return output
 
  
if __name__ == "__main__":
  # TESTS
  initialize()
  print("initialization... Done!")
  # lda = getLDA()
  # test_text = cleantext('Pajama Sam in No Need to Hide When it\'s Dark Outside')

  # test_vec = dictionary.doc2bow(tokenize(test_text))

  # vec_lda = lda[test_vec]
  # sims = similarity_index[vec_lda]
  # sims = sorted(enumerate(sims), key = lambda item: -item[1])
  # print(np.asarray(data_frame.iloc[sims[0][0]]))
  # print(tokens[0])
  

  