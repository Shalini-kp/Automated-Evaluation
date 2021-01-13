import gensim
from gensim import corpora
from pprint import pprint

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet # Lemmatize with POS Tag
from textblob import TextBlob
from nltk.corpus import stopwords

# How to create a dictionary from a list of sentences?
documents = '''An entrepreneur is an individual who creates a new business, bearing most of the risks and enjoying most of the rewards. The entrepreneur is commonly seen as an innovator, a source of new ideas, goods, services, and business/or procedures.
Entrepreneurs play a key role in any economy, using the skills and initiative necessary to anticipate needs and bring good new ideas to market. Entrepreneurs who prove to be successful in taking on the risks of a startup are rewarded with profits, fame, and continued growth opportunities.
Those who fail, suffer losses and become less prevalent in the markets.'''
print(documents)
print("*************************")

# Tokenize: Split the paragraph into sentences
#sentences = nltk.sent_tokenize(documents)

# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(documents)
stop_words = set(stopwords.words('english'))

#need to check
#word_list = [w for w in word_list if not w in stop_words]
print(word_list)
print("*************************")

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    # tag = nltk.pos_tag([word])[0][1][0].upper()
    # tag_dict = {"J": wordnet.ADJ,
    #             "N": wordnet.NOUN,
    #             "V": wordnet.VERB,
    #             "R": wordnet.ADV}
    #
    # return tag_dict.get(tag, wordnet.NOUN)
    sent = TextBlob(word)
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    words_and_tags = [(w, tag_dict.get(pos[0], wordnet.NOUN)) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)


# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

#  lemmatized_output = get_wordnet_pos(i)
# Lemmatize list of words and join
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)
print("*************************")

#Using FastText
from gensim.models import FastText
modelWords_FastText = FastText(lemmatized_output, size=100, window=5, min_count=5, workers=4,sg=1)
#modelWords_FastText.build_vocab(sentences=lemmatized_output)
modelWords_FastText.train(sentences=lemmatized_output, total_examples=len(lemmatized_output), epochs=10)  # train
print(modelWords_FastText.wv)
print("*************************")

#Using Word2Vec
from gensim.models import Word2Vec
#modelWords_Word2Vec = Word2Vec(sentences=lemmatized_output, size=100, window=5, min_count=5, workers=4, sg=0)
#print((modelWords_Word2Vec))

# Tokenize the sentences into words using split
#texts = [text for text in documents.split()]
#print("*************************")

# Create dictionary
#dictionary = corpora.Dictionary(texts)

# Get information about the dictionary
#print(dictionary.token2id)
