import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import torch
import numpy as np

class WordProcessor(object):
    def __init__(self,corpus):
        self.corpus = corpus

    def preprocess(self):
        stop_words = set(stopwords.words('english'))
        data = []
        senteces = self.corpus.split('.')
        for i in senteces:
            sentence = i.strip().split()
            x = [word.strip(string.punctuation) for word in sentence if word not in stop_words]
            x = [word.lower() for word in x]
            data.append(x)
        return data

    def get_vocab(self, sentences):

        data = {}
        vocab = {}
        for s in sentences:
            for word in s: 
                if word not in data:
                    data[word] = 1
                else:
                    data[word] += 1

        data = sorted(list(data.keys()))
        for i in range(len(data)):
            vocab[data[i]] = i

        return vocab


    def skip_gram(self, sentences, context_size):

        X = []
        y = []
        
        vocab = self.get_vocab(sentences)
        V = len(vocab)

        for s in sentences:
            for i in range(len(s)):
                center_word = np.zeros(V)
                center_word[vocab[s[i]]] = 1

                for neighbour in range(max(0,i - context_size),min(len(s),i + context_size + 1 )):
                    if i != neighbour:

                        context = np.zeros(V)
                        context[vocab[s[neighbour]]] = 1
                        X.append(center_word)
                        y.append(context)

        return np.array(X),np.array(y), vocab

    def cbow(self, sentences, context_size):
        X = []
        y = []
        
        vocab = self.get_vocab(sentences)
        V = len(vocab)
        for s in sentences:
            for i in range(len(s)):
                target = np.zeros(V)
                target[vocab[s[i]]] = 1
                context = np.zeros(V)

                n_neighbours = 0
                for neighbour in range(max(0,i - context_size),min(len(s),i + context_size + 1)):
                    if i != neighbour:
                        context[vocab[s[neighbour]]] += 1
                        n_neighbours += 1

                if n_neighbours > 0: 
                    context = context / n_neighbours

                X.append(context)  
                y.append(target)

        return np.array(X),np.array(y)  
                
               

        

                    



                    
                
        
