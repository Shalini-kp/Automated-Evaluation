#!usr/bin/python3

import nltk
sentence = "I want to check if a sentence has a specific parts of speech tag structure."
tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
grammar = r"""
NP: 
{<NNS><IN><NN><NN><NN>}
{<PRP><VBP>}
"""

cp = nltk.RegexpParser(grammar)
result = cp.parse(tagged)
print(result)