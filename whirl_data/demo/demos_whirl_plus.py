"""
Whirl ProbLog implementation in animal dataset

Builds a TF-IDF model in the background using NLTK and Scikit-learn, such
that ProbLog can make use of document similarity while reasoning.

From:
W. W. Cohen. Whirl: A word-based information representation language.
Artificial Intelligence, 118(1):163-196, 2000.

Author:
- Wannes Meert
- Anton Dries
Modified by Hikaru
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


#sys.path.append('../../')
sys.path.append(os.path.abspath('F:\\Program Files\\Anaconda\\Lib\\site-packages\\'))

from problog.extern import problog_export_nondet


# Settings
texts = glob.glob('E:\\GitHub\\document-soft-join-in-problog\\whirl_data\\demo\\*.txt') # input texts for TF-IDF model
filePath = "E:\\Download\\chrome download\\problog2.1\\whirl_data\\demo\\"
baseFile = "demos_base.dic"

# Global variables
vectorizer = None
tfidf = None # TF-IDF model
tokens = None # List of tokens used by TF-IDF
texts_weights = None # TF-IDF weights for texts
texts_dic = {} # Dictionary to store all texts
texts_key = {} # Dictionary to store all texts_key pairs of base case
num_clusters = 28 # Cluster for K-Means
n_of_using_cluster = 3 # should <= sqrt(num_clusters)
data_frame = None
km = None

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
  tokens = nltk.word_tokenize(text)
  stemmed = (stemmer.stem(token) for token in tokens)
  return stemmed


def getTFIDF():
  """Return cached TFIDF model."""
  global vectorizer
  global tfidf
  global tokens
  global texts_weights
  global texts_dic
  global texts_key

  if tfidf is None:

    texts_content = []
    for text in texts:
      with open(text, 'r') as ifile:
        for line in ifile:
          line1 = line[:-1]
          texts_content.append(cleantext(line1))
          if not line1 in texts_dic:
            texts_dic[line1] = cleantext(line1)
    with open((filePath + baseFile), 'r') as dicfile:
      for line in dicfile:
        line1 = re.split('\t', line)
        line2 = line1[2][:-1]
        texts_key[line2] = line1[1]

    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english')
    texts_counts = vectorizer.fit_transform(texts_content)
    tokens = vectorizer.get_feature_names() 
    
    tfidf = TfidfTransformer()
    # fit_transform can be replaced by fit for efficiency
    texts_weights = tfidf.fit_transform(texts_counts)
    print(np.shape(texts_weights))
    texts_weights1 = texts_weights
    print(np.histogram(texts_weights1.toarray()))
    transformWords()
    if (not os.path.isfile(filePath + 'doc_cluster.pkl')) and (not os.path.isfile(filePath + 'frame.pkl')):
      kmeansClusting(num_clusters, texts_weights)
  return tfidf

def transformWords():
  """Transform all words in database into vector"""
  global texts_dic
  if texts_dic:
    tfidf = getTFIDF()
    for key in texts_dic.keys():
      texts_dic[key] = tfidf.transform(vectorizer.transform([texts_dic[key]]))

def kmeansClusting(nc, tfidfMatrix):
  """K-Means given number of clusters"""
  global km
  global data_frame
  km = KMeans(n_clusters = nc)
  km.fit(tfidfMatrix)
  clusters = km.labels_.tolist()
  joblib.dump(km, (filePath + 'doc_cluster.pkl'))
  com_name = []
  clusters = km.labels_.tolist()
  for text in texts:
      with open(text, 'r') as ifile:
        for line in ifile:
          com_name.append(line[:-1])

  datasets = {'commonName': com_name, 'cluster':clusters}
  data_frame = pd.DataFrame(datasets, index = [clusters], columns = ['commonName', 'cluster'])
  joblib.dump(data_frame, (filePath + 'frame.pkl'))

def compareCluster(v2, km):
  """comparing similarity of vector with central vector of all clusters and return with order of similarity from high to low (not return real value)"""
  cluster_central_sim_arr = [[0, 0]] * num_clusters   #second number is the order of cluster, first is the similarity score
  for i in range(num_clusters):
    cluster_central_sim_arr[i] = [cosine_similarity(v2, km.cluster_centers_[i])[0, 0], i]
  return sorted(cluster_central_sim_arr, reverse = True)  #return the order of similarity descendingly

def bestMatch(v2, n_of_using_cluster, clus_sim):
  """compare v with all given number of clusters"""
  highest_sim = 0
  most_sim_texts = []
  for i in range(n_of_using_cluster):
    if clus_sim[i][0] > 0:
      #print(type(data_frame))
      if (type(data_frame.ix[clus_sim[i][1]]['commonName']) == type("")):
        cn = data_frame.ix[clus_sim[i][1]]['commonName']
        if cn in texts_dic.keys():
          cnv = texts_dic[cn]
        else:
          cnv = tfidf.transform(vectorizer.transform([cleantext(cn)]))
          texts_dic[cn] = cnv
        text_sim_test = cosine_similarity(cnv, v2)
        if text_sim_test[0,0] > highest_sim:
          highest_sim = text_sim_test[0,0]
          most_sim_texts = [(cn, texts_key[cn], float('%.4f'% highest_sim))]
          if highest_sim - 1 == 0:
            return most_sim_texts
        elif text_sim_test[0, 0] == highest_sim:
          most_sim_texts.append((cn, texts_key[cn], float('%.4f'% highest_sim)))
      else:
        for cn in data_frame.ix[clus_sim[i][1]]['commonName'].values.tolist():
          #print(cn)
          if cn in texts_dic.keys():
            cnv = texts_dic[cn]
          else:
            cnv = tfidf.transform(vectorizer.transform([cleantext(cn)]))
            texts_dic[cn] = cnv
          text_sim_test = cosine_similarity(cnv, v2)
          if text_sim_test[0,0] > highest_sim:
            highest_sim = text_sim_test[0,0]
            most_sim_texts = [(cn, texts_key[cn], float('%.4f'% highest_sim))]
            if highest_sim - 1 == 0:
              #print(most_sim_text)
              return most_sim_texts
          elif text_sim_test[0, 0] == highest_sim:
            most_sim_texts.append((cn, texts_key[cn], float('%.4f'% highest_sim)))
    else:
      break
  if highest_sim > 0.5:
    return most_sim_texts
  else:
    return [('None', 'None', 0)]

# def levenshtein(a, b):
#    """Calculates the Levenshtein distance between a and b."""
#   n, m = len(a), len(b)
#   if n > m:
#       # Make sure n <= m, to use O(min(n,m)) space
#       a,b = b,a
#       n,m = m,n
      
#   current = range(n+1)
#   for i in range(1,m+1):
#       previous, current = current, [i]+[0]*n
#       for j in range(1,n+1):
#           add, delete = previous[j]+1, current[j-1]+1
#           change = previous[j-1]
#           if a[j-1] != b[i-1]:
#               change = change + 1
#           current[j] = min(add, delete, change)
          
#   return current[n]

# def bestLevenshtein(e2, result_list):
#   best_result = None
#   best_score = 0
#   for i in result_list:
#     current_score = levenshtein(e2, i[0])
#     if current_score > best_score:
#       best_result = [i]
#     elif current_score == best_score:
#       best_result.append(i)
#   return best_result

@problog_export_nondet('+str', '-list')  #for similarity(Str1, Str2, P)
def similarity(e2):
  """TF-IDF similarity between two documents based on pre-processed texts.
     Expects two text/document encodings.
     Input: one query string.
     Output: identical string and similarity
  """
  global texts_dic
#  global km
  global data_frame

  most_sim_clus = 0
  clus_sim = 0
  most_sim_text = None
  text_sim = 0

  tfidf = getTFIDF()
  #if (not km) and (not data_frame):
  km = joblib.load(filePath + 'doc_cluster.pkl')
  data_frame = joblib.load(filePath + 'frame.pkl')
  clusters = km.labels_.tolist()
  if e2 in texts_dic.keys():
    v2 = texts_dic[e2]
  else:
    v2 = tfidf.transform(vectorizer.transform([cleantext(e2)]))
    texts_dic[e2] = v2
  
  # for i in range(num_clusters):  
  #   clus_sim_test = cosine_similarity(v2, km.cluster_centers_[i])
  #   if clus_sim_test[0,0] > clus_sim:
  #     clus_sim = clus_sim_test[0,0]
  #     most_sim_clus = i
  
  clus_sim = compareCluster(v2, km) #list of list [[sim, order_of_cluster]] similarity from high to low

  if clus_sim[0][0] < 0.001:  # not similar to any cluster center vector
    return [('None', 'None', 0)]
  else:
    output0 = bestMatch(v2, n_of_using_cluster, clus_sim)
    return output0
    # if len(output0) == 1:
    #   return output0
    # else:
    #   return bestLevenshteine(e2, output0)
    # for cn in data_frame.ix[most_sim_clus]['commonName'].values.tolist():
    #   #print(cn)
    #   if cn in texts_dic.keys():
    #     cnv = texts_dic[cn]
    #   else:
    #     cnv = tfidf.transform(vectorizer.transform([cleantext(cn)]))
    #     texts_dic[cn] = cnv
    #   text_sim_test = cosine_similarity(cnv, v2)
    #   if text_sim_test[0,0] > text_sim:
    #     text_sim = text_sim_test[0,0]
    #     most_sim_text = cn
    #     if text_sim - 1 == 0:
    #       #print(most_sim_text)
    #       return [(most_sim_text, texts_key[most_sim_text], 1)]
    # if text_sim > 0.5:
    #   return [(most_sim_text, texts_key[most_sim_text], float('%.4f'% text_sim))]
    # else:
    #   return [('None', 'None', 0)]
  
if __name__ == "__main__":
  # TESTS
  model = getTFIDF()
  #kmeansClusting(num_clusters, texts_weights)
  km = joblib.load(filePath + 'doc_cluster.pkl')
  data_frame = joblib.load(filePath + 'frame.pkl')

  