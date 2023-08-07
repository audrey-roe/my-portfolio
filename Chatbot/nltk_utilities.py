import nltk
import numpy as np
# nltk.download('punkt') # this is a package with a pre-trained tokenizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower()) # to stem and convert to lower case letters

def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hi", "there", "hi", "mabi", "hello","haroo","yaw","wassup", "hey", "holla", "hello"]
    words = ["hello",  "thanks", "for", "checking", "in", "hi", "there", "how", "mabi" "can", "i", "help", you"]
    bag =   [0,         0,          0,          0,   0,     1,      1,      0,      0,    0,  0,      0,  0]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words): # idx=index... looping over all words
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

